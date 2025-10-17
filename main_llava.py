import os
from pathlib import Path
import logging
import argparse
import time
import pynvml
from tqdm import tqdm
# from scipy import log
from numpy import log
import numpy as np
from PIL import Image
import json
import torch
from lavis.models import load_model_and_preprocess
import cv2
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_adversarial_noise(x, epsilon=0.1):
    noise = torch.zeros_like(x, requires_grad=True)
    return noise



@torch.no_grad()
def measure_gpu(model, inputs, iter_num, device_id, tokenizer):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    multi_cap_list = []
    t1 = time.time()
    power_list = []
    avg_length, verbose_length, verbose_caption = 0, 0, 0
    for _ in range(iter_num):
        caption = model.generate(
                inputs["prompt"][0],
                images = inputs["image"].half(),
                num_beams=1,
                do_sample=True,
                max_new_tokens=512,
                output_attentions=True,
                top_p=0.9, 
                temperature=1,
                use_cache=True,
                repetition_penalty=1
            )
        
        input_token_len = inputs["prompt"][0].shape[1]
        outputs = tokenizer.batch_decode(caption[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        length = len(outputs.split(' '))
        multi_cap_list.append(outputs)
        avg_length += (length / iter_num)
        if length > verbose_length:
            verbose_length = length
            verbose_caption = outputs
        power = 0
        power_list.append(power)
    t2 = time.time()
    latency = (t2 - t1) / iter_num
    s_energy = sum(power_list) / len(power_list) * latency
    energy = s_energy / (10 ** 3) / iter_num        
    pynvml.nvmlShutdown()
    print(outputs)
    print('avg_length={}, latency={}, energy={}'.format(avg_length, latency, energy))
    return latency, energy, verbose_caption, avg_length, multi_cap_list


class TestModel():
    def clamp(self, delta, clean_imgs):
        MEAN = torch.tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]]).to(clean_imgs.device)
        STD = torch.tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]]).to(clean_imgs.device)

        clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
        clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data

        return clamp_delta


    def test_model(self, args, logger):

        # Model
        # disable_torch_init()
        model_path = os.path.expanduser(args.model_path)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=True,)
        model = model.eval()
        model.enable_input_require_grads()

        ITER, STEP_SIZE, EPSILON = args.iter, args.step_size, args.epsilon
        # input_text = "Question: Describe the details of the given image. Answer:"
        # input_text = ""
        input_text = "Describe the details of the given image."
        
        if model.config.mm_use_im_start_end:
            input_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + input_text
        else:
            input_text = DEFAULT_IMAGE_TOKEN + '\n' + input_text
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


        filenames = os.listdir('imgs/sub_val')
        image_filenames = [filename for filename in filenames if filename.split('.')[-1].lower() in ['jpg']]
        total_average_len, total_average_ori_len, sample_num = 0, 0, len(image_filenames)
        for filename in image_filenames:
            raw_image = Image.open("imgs/sub_val/"+filename)
            image = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
            image = image.half().to(device)
            delta = torch.randn_like(image, device=device, requires_grad= True, dtype=image.dtype).half()
            delta = create_adversarial_noise(delta)
            
            verbose_len, verbose_energy, verbose_latency = 0, 0, 0
            ori_latency, ori_energy, ori_caption, ori_len, ori_multi_cap_list = measure_gpu(model, {"image": image.half(), "args": args, "prompt": [input_ids], "logger": logger}, 3, args.gpu, tokenizer)
            print('original text length={}'.format(ori_len))
            print('original text: {}'.format(ori_caption))
            output_text = ori_caption
            
            verbose_multi_cap_list = []
            verbose_len_list, verbose_energy_list, verbose_latency_list, ori_latency_list, ori_energy_list, ori_len_list = [],[],[],[],[],[]
            ori_latency_list.append(ori_latency)
            ori_energy_list.append(ori_energy)
            ori_len_list.append(ori_len)

            prompt = conv.get_prompt()
            # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            for tdx in range(ITER):
                result = model.generate_verbose_images(
                                input_ids = input_ids,
                                images = (image + delta).half(),
                                output_attentions=True,
                            )      
                # result = model.generate_verbose_images({"image": image + delta, "text_input": [input_text + '' + output_text], "logger": logger})
                loss1, loss2, loss3 = result["loss1"], result["loss2"], result["loss3"]

                print('iter={}, loss1={}, loss2={} loss3={}'.format(tdx, loss1, loss2, loss3))
    ######################################################################
                loss = loss1 + loss2 + loss3  # only use loss1
    ######################################################################

                model.zero_grad()
                loss.backward(retain_graph=False)
                delta.data = delta.data - STEP_SIZE * torch.sign(delta.grad.detach())
                delta.data = self.clamp(delta, image).clamp(-EPSILON, EPSILON)
                delta.grad.zero_()
                            
                output_latency, output_energy, output_text, output_len, output_multi_cap_list = measure_gpu(model, {"image": (image + delta).half(), "args": args, "prompt": [input_ids], "logger": logger}, 3, args.gpu, tokenizer)

                if output_len > verbose_len:
                    verbose_energy = output_energy
                    verbose_latency = output_latency
                    verbose_len = output_len
                    verbose_multi_cap_list = output_multi_cap_list
                    print('------------------------------------------------')
                    print('best_length_till_now={}'.format(verbose_len))
                    print('------------------------------------------------')
                    
            verbose_latency_list.append(verbose_latency)
            verbose_energy_list.append(verbose_energy)
            verbose_len_list.append(verbose_len)

            logger.info('Original images, Length: %.2f, Energy: %.2f, Latency: %.2f', ori_len, ori_energy, ori_latency)
            logger.info('Verbose images, Length: %.2f, Energy: %.2f, Latency: %.2f', verbose_len, verbose_energy, verbose_latency)

            total_average_ori_len += ori_len
            total_average_len += verbose_len

        logger.info('!!!!Original Average, Length!!!!!: %.2f', (total_average_ori_len/args.test_sample))
        logger.info('!!!!After Average, Length!!!!!: %.2f', (total_average_len/args.test_sample))
        


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('generate verbose images')
    parser.add_argument("--model-path", type=str, default="/home/hfs/e/llm/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument('--epsilon', type=float, default=0.032, help='the perturbation magnitude')
    parser.add_argument('--step_size', type=float, default=0.0039, help='the step size')
    parser.add_argument('--iter', type=int, default=500, help='the iteration')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    parser.add_argument('--root_path', type=str, default='results')
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument("--test_sample", type=int, default=500)
    return parser.parse_args()


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    exp_dir = Path(os.path.join(args.root_path, 'log'))
    exp_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath(args.dataset)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("OPT")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
        
    test_blip2 = TestModel()
    test_blip2.test_model(args, logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)

