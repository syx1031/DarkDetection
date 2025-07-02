import torch
import os
import argparse
import random
import numpy as np
from CLIP4Clip.modules.modeling import CLIP4Clip
from CLIP4Clip.dataloaders.rawvideo_util import RawVideoExtractor
from torch.utils.data import Dataset, DataLoader
import json

from utils import time_to_seconds


class My_DataLoader(Dataset):
    def __init__(
            self,
            data_path,
            features_path,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
    ):
        self.data_path = data_path
        self.features_path = features_path
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = 0
        self.max_frames = max_frames

        original_features_dict = json.load(open(features_path, "r", encoding="utf-8"))

        self.features_dict = []
        for feature in original_features_dict:
            for start_time, end_time in zip(feature["starts"], feature["ends"]):
                self.features_dict.append({
                    "path": feature["path"],
                    "starts": [start_time],
                    "ends": [end_time]
                })

        self.video_dict = [None] * len(self.features_dict)
        for idx, data in enumerate(self.features_dict):
            self.video_dict[idx] = os.path.join(data_path, data["path"])

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.features_dict)

    def _get_rawvideo(self, idx, s, e):
        video_mask = np.zeros((len(s), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(s)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(s), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=float)
        video_path = self.video_dict[idx]

        for i in range(len(s)):
            start_time = time_to_seconds(s[i])
            end_time = time_to_seconds(e[i])

            cache_id = "{}_{}_{}".format(video_path, start_time, end_time)
            # Should be optimized by gathering all asking of this video
            raw_video_data = self.rawVideoExtractor.my_get_video_data(video_path, start_time, end_time, self.max_frames)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)

                video_slice = raw_video_slice

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}, start: {}, end: {}".format(video_path, idx, start_time, end_time))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, feature_idx):
        feature = self.features_dict[feature_idx]
        starts = feature["starts"]
        ends = feature["ends"]

        video, video_mask = self._get_rawvideo(feature_idx, starts, ends)
        return video, video_mask


def dataloader_mydata_test(args):
    mydt_testset = My_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        feature_framerate=args.feature_framerate,
    )
    dataloader_mydt = DataLoader(
        mydt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_mydt, len(mydt_testset)


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True, help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='E:\\DarkDetection\\dataset\\syx\\us', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='local_database\\video.json', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default="CLIP4Clip/result", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def set_seed_logger(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # world_size = 1
    # torch.cuda.set_device(args.local_rank)
    # args.world_size = world_size
    # rank = torch.distributed.get_rank()
    # args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def init_model(args, device):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model


def _run_on_single_gpu(model, batch_list_v, batch_visual_output_list):
    sim_embedding = []
    for idx2, b2 in enumerate(batch_list_v):
        video_mask, *_tmp = b2
        visual_output = batch_visual_output_list[idx2]
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        b1b2_embeddings = model.loose_embedding(visual_output, video_mask, sim_header="meanP")
        b1b2_embeddings = b1b2_embeddings.cpu().detach().numpy()
        sim_embedding.append(b1b2_embeddings)
    return sim_embedding


def get_video_embedding():
    args = get_args()
    args = set_seed_logger(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(args, device)

    test_dataloader, test_length = dataloader_mydata_test(args)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        batch_list_v = []
        batch_visual_output_list = []

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            video, video_mask = batch

            visual_output = model.get_visual_output(video, video_mask)

            batch_visual_output_list.append(visual_output)
            batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        embeddings = _run_on_single_gpu(model, batch_list_v, batch_visual_output_list)
        pass


if __name__ == "__main__":
    get_video_embedding()
