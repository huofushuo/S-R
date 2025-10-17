import os
from pathlib import Path
import logging
import argparse
import time
from tqdm import tqdm
from numpy import log
import numpy as np
from PIL import Image
import pynvml
import torch
import random
from lavis.models import load_model_and_preprocess
import cv2
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils
# setup device to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




@torch.no_grad()
def measure_gpu(model, input_token, iter_num, device_id, temp):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    multi_cap_list = []
    t1 = time.time()
    power_list = []
    avg_length, verbose_length, verbose_caption = 0, 0, 0
    for _ in range(iter_num):
        caption = model.generate(input_token, use_nucleus_sampling=True, top_p=0.9, temperature=1, num_beams=1, max_length=512, repetition_penalty=1)[0]
        length = len(caption.split(' '))
        multi_cap_list.append(caption)
        avg_length += (length / iter_num)
        if length > verbose_length:
            verbose_length = length
            verbose_caption = caption
        power = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_list.append(power)
    t2 = time.time()
    latency = (t2 - t1) / iter_num
    s_energy = sum(power_list) / len(power_list) * latency
    energy = s_energy / (10 ** 3) / iter_num               
    pynvml.nvmlShutdown()
    print('length={}, latency={}, energy={}'.format(avg_length, latency, energy))
    return latency, energy, verbose_caption, avg_length, multi_cap_list






class TestModel:
    def clamp(self, delta, clean_imgs):
        MEAN = torch.tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]]).to(clean_imgs.device)
        STD = torch.tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]]).to(clean_imgs.device)

        clamp_imgs = (((delta.data + clean_imgs.data) * STD + MEAN) * 255).clamp(0, 255)
        clamp_delta = (clamp_imgs/255 - MEAN) / STD - clean_imgs.data

        return clamp_delta


    def test_model(self, args, logger):

        if args.test_model_name == 'blip2_opt2.7b':
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
            )
        elif args.test_model_name == 'blip2_opt6.7b':
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
            )
        elif args.test_model_name == 'blip2_vicuna7b_instruct':
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device
            )

        model = model.eval()

        ITER, STEP_SIZE, EPSILON = args.iter, args.step_size, args.epsilon
        if args.model_name == 'blip2_opt2.7b' or args.model_name == 'blip2_opt6.7b':
            input_text = "Question: Describe the details of the given image. Answer:"
        else:
            input_text = "Describe the details of the given image."

        filenames = os.listdir('./imgs/sub_val')
        image_filenames = [filename for filename in filenames if filename.split('.')[-1].lower() in ['jpg']]
        total_average_len, total_average_ori_len, sample_num = 0, 0, len(image_filenames)
        for filename in image_filenames:
            raw_image = Image.open("imgs/sub_val/"+filename).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            print('------------------------------------------------')
            print('current file name:{}'.format(filename))
            print('------------------------------------------------')
        
            image = image.to(device)
            delta = torch.randn_like(image, requires_grad=True)
            verbose_len, verbose_latency = 0, 0
            ori_latency, ori_energy, ori_caption, ori_len, ori_multi_cap_list = measure_gpu(model, {"image": image, "args": args, "prompt": [input_text], "logger": logger}, 3, args.gpu, temp=1)
            print('original text length={}'.format(ori_len))
            print('original text: {}'.format(ori_caption))
            output_text = ori_caption
            
            verbose_multi_cap_list = []
            verbose_len_list, verbose_latency_list, ori_latency_list, ori_len_list = [],[],[],[]
            ori_latency_list.append(ori_latency)
            ori_len_list.append(ori_len)

            for tdx in range(ITER):
                result = model.generate_verbose_images({"image": image + delta, "text_input": [output_text], "logger": logger})
                # result = model.generate_verbose_images({"image": image + delta, "text_input": [input_text + '' + output_text], "logger": logger})
                loss1, loss2, loss3 = result["loss1"], result["loss2"], result["loss3"]
                print('iter={}, loss1={}, loss2={} loss3={}'.format(tdx, loss1, loss2, loss3))
                loss = loss1 + loss2 + loss3   # only use loss1

                model.zero_grad()
                loss.backward(retain_graph=False)
                delta.data = delta - STEP_SIZE * torch.sign(delta.grad.detach())
                delta.data = self.clamp(delta, image).clamp(-EPSILON, EPSILON)
                delta.grad.zero_()
                            
                output_latency, ori_energy,output_text, output_len, output_multi_cap_list = measure_gpu(model, {"image": image + delta, "args": args, "prompt": [input_text], "logger": logger}, 1, args.gpu, temp=1)

                if output_len > verbose_len:
                    # temp_delta1 = delta
                    verbose_latency = output_latency
                    verbose_len = output_len
                    verbose_multi_cap_list = output_multi_cap_list
                    print('------------------------------------------------')
                    print('best_length_till_now={}'.format(verbose_len))
                    print(output_text)
                    print('------------------------------------------------')
                    
                    
            verbose_latency_list.append(verbose_latency)
            verbose_len_list.append(verbose_len)

            logger.info('Image name: %s', filename)
            logger.info('Original images, Length: %.2f, Latency: %.2f', ori_len, ori_latency)
            logger.info('Verbose images, Length: %.2f, Latency: %.2f', verbose_len, verbose_latency)

            total_average_ori_len += ori_len
            total_average_len += verbose_len

        logger.info('!!!!Original Average, Length!!!!!: %.2f', (total_average_ori_len/sample_num))
        logger.info('!!!!After Average, Length!!!!!: %.2f', (total_average_len/sample_num))
        


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('generate verbose images')
    parser.add_argument('--epsilon', type=float, default=0.032, help='the perturbation magnitude')
    parser.add_argument('--step_size', type=float, default=0.0039, help='the step size')
    parser.add_argument('--iter', type=int, default=500, help='the iteration')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--seed', type=int, default=256, help='random seed')
    parser.add_argument('--root_path', type=str, default='results', help='noise; noise+image')
    parser.add_argument('--dataset', type=str, default='MSCOCO')
    parser.add_argument("--model_name", type=str, default='blip2_opt6.7b', help='attacked model name (blip2_opt2.7b, blip2_opt6.7b, blip2_vicuna7b_instruct)')
    parser.add_argument("--test_model_name", type=str, default='blip2_opt6.7b', help='inference (test) model name (blip2_opt2.7b, blip2_opt6.7b, blip2_vicuna7b_instruct)')
    parser.add_argument("--test_sample", type=str, default='test_sample5', help='tested sample name (test_sample1-5)')
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    from lavis.models import model_zoo
    # print(model_zoo)
    
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
        
    test_blip2 = TestModel()
    test_blip2.test_model(args, logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)

