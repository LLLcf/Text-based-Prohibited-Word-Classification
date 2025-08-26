```
# cuda 12.1
conda create -n nlp_env python=3.10 -y
conda activate nlp_env
pip install ipykernel
python -m ipykernel install --name nlp_env
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install pandas numpy torch torchvision torchaudio transformers==4.43 scikit-learn tqdm
# 运行四个训练脚本
python BERT_FGM_focal_loss.py
python BERT_FGM_Rdrop-multigpu.py
python BERT_FGM_Rdrop_focal_loss-multigpu.py
python BERT_FGM_Rdrop_with_label_smooth-multigpu.py
# 集成推理脚本
python inference.py
# 二次集成脚本
# 运行 预测结果分析.ipynb
```