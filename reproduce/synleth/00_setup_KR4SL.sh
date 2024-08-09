#!/bin/bash
#SBATCH --job-name=setup_KR4SL
#SBATCH --output=./slurm_out/setup_KR4SL.out
#SBATCH --error=./slurm_out/setup_KR4SL.err
#SBATCH --time=12:00:00
#SBATCH --mem=64Gb
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal

# To run this file:
# cd ./reproduce/synleth/
# sbatch ./scripts/00_setup_KR4SL.sh

if [ ! -d "KR4SL/" ]; then
  git clone https://github.com/JieZheng-ShanghaiTech/KR4SL
fi

## Set-up environment:

if [ ! -d "miniconda3/" ]; then
  mkdir -p miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
  bash miniconda3/miniconda.sh -b -u -p miniconda3
  rm -rf miniconda3/miniconda.sh
fi

if [ ! -d "miniconda3/envs/synleth_env/" ]; then
  conda create -n synleth_env python=3.9
  conda activate synleth_env
  conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
  conda install pytorch-scatter -c pyg
  conda install -c conda-forge transformers
  conda install -c anaconda scipy
  conda install -c conda-forge ipdb gensim
  conda install scikit-learn
  conda deactivate
fi


## Extract textual embeddings:

source miniconda3/etc/profile.d/conda.sh
conda activate synleth_env

export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/targets/x86_64-linux/lib

if [ ! -f "KR4SL/data/all_entities_pretrain_emb.npy" ]; then
  cd KR4SL/
  mv data/extract_pretrain_emb.py ./
  mv data/all_entities.txt data/all_entities
  python extract_pretrain_emb.py
  mv all_entities_pretrain_emb.npy data/
fi


## Prepare for training:

MODELS_FILE="KR4SL/transductive/models.py"
# Change line 95
#sed -i '95s/edges\[:,1\]\].cuda()/edges[:,1].cpu()].cuda()/; 95s/edges\[:,3\]\].cuda()/edges[:,3].cpu()].cuda()/' "$MODELS_FILE"
# Change line 107
#sed -i '107s/entity_pretrain_emb\[q_sub\[nodes\[:,0\]\]\].cuda()/entity_pretrain_emb[q_sub[nodes[:,0]].cpu()].cuda()/' "$MODELS_FILE"

BASE_MODEL_FILE="KR4SL/transductive/base_model.py"
sed -i '/from operator import itemgetter/a import os' "$BASE_MODEL_FILE"
sed -i '/self\.explain = args\.explain/a \ \ \ \ \ \ \ \ self.setting = args.setting' "$BASE_MODEL_FILE"
sed -i '/self\.explain = args\.explain/a \ \ \ \ \ \ \ \ self.suffix = args.suffix' "$BASE_MODEL_FILE"
sed -i "/filters_obj = self\.loader\.filters/a \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ fp.write(\"%s\\\\n\" % item)" "$BASE_MODEL_FILE"
sed -i "/filters_obj = self\.loader\.filters/a \ \ \ \ \ \ \ \ \ \ \ \ for item in gene_idx:" "$BASE_MODEL_FILE"
sed -i "/filters_obj = self\.loader\.filters/a \ \ \ \ \ \ \ \ with open(os.path.join('../results/'+self.suffix, self.setting, 'gene_idx_'+str(self.seed)+'.txt'), 'w') as fp:" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ fp.write(\"%s\\\\n\" % item)" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ for item in r_all:" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ with open(os.path.join('../results/'+self.suffix, self.setting, 'r_all_seed_'+str(self.seed)+'.txt'), 'w') as fp:" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ np.save(os.path.join('../results/'+self.suffix, self.setting, 'f_all_seed_'+str(self.seed)+'.npy'), f_all)" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ np.save(os.path.join('../results/'+self.suffix, self.setting, 's_all_seed_'+str(self.seed)+'.npy'), s_all)" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ fp.write('\\\\n'.join('%s %s' % x for x in self.loader.test_q))" "$BASE_MODEL_FILE"
sed -i "/# f_all = np\.vstack(f_all)/a \ \ \ \ \ \ \ \ \ \ \ \ with open(os.path.join('../results/'+self.suffix, self.setting, 'test_q_'+str(self.seed)+'.txt'), 'w') as fp:" "$BASE_MODEL_FILE"

TRAIN_FILE="KR4SL/transductive/train.py"
sed -i "/parser\.add_argument('--time_num', type=int, default=0)/a \ \ \ \ parser.add_argument('--suffix', type=str, default='trans')\n\ \ \ \ parser.add_argument('--setting', type=str, default='epochs15_noDropout')" "$TRAIN_FILE"
sed -i "/parser\.add_argument('--seed', type=str, default=1234)/s/type=str/type=int/" "$TRAIN_FILE"
sed -i "/torch\.cuda\.set_device(args\.gpu)/i \ \ \ \ opts.perf_file = os.path.join('../results/'+suffix, args.setting, suffix+'_perf_seed'+str(args.seed)+'_'+str(args.time_num)+'fold'+'.txt')" "$TRAIN_FILE"
sed -i "/torch\.cuda\.set_device(args\.gpu)/i \ \ \ \ opts.ckpt_file = os.path.join('../results/'+suffix, args.setting, suffix+'_ckpt_model_seed'+str(args.seed)+'_'+str(args.time_num)+'fold'+'.pkl')" "$TRAIN_FILE"
sed -i "/torch\.cuda\.set_device(args\.gpu)/i \ \ \ \ opts.best_1fold_file = os.path.join('../results/'+suffix, args.setting, suffix+'_best_seed'+str(args.seed)+'_'+str(args.time_num)+'fold_model'+'.pkl')" "$TRAIN_FILE"
sed -i "/torch\.cuda\.set_device(args\.gpu)/i \ \ \ \ os.environ['CUDA_VISIBLE_DEVICES'] = '0'" "$TRAIN_FILE"
sed -i "/opts\.n_rel = loader\.n_rel/a \ \ \ \ opts.seed = args.seed\n\ \ \ \ opts.setting = args.setting" "$TRAIN_FILE"
sed -i "s|np.save('results/'+suffix+'/all_result_'+str(time_num)+'fold.npy', all_result)|np.save(os.path.join('../results/'+suffix, args.setting, 'all_result_seed_'+str(args.seed)+'_'+str(args.time_num)+'fold.npy'), all_result)|g" "$TRAIN_FILE"
sed -i "/opts\.n_rel = loader\.n_rel/a \ \ \ \ opts.suffix = args.suffix" "$TRAIN_FILE"