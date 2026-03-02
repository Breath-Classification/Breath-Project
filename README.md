# Breath-Project
xhost +local:docker
pip install -e .  -uzywane aby byly zaleznosci miedzy plikami


## Tworzenie venva graficznego
python -m venv venv_streamlit --system-site-packages
source venv_streamlit/bin/activate
pip install --upgrade pip
pip install streamlit
streamlit run local_app.py --server.address=0.0.0.0 --server.port=8501
ctr+c 
deactivate

1) GRU
2) LSTM
3) stacked LSTM 2 warstwy
4) CNN+LSTM
5) Attention LSTM
6) Bidirectional LSTM

pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

✓ sliding window feature extraction
✓ temporal windowing
✓ frame stacking
✓ block-wise representation
✓ context windows
✓ local temporal embedding
✓ multi-step input for RNN