# News Recommendation

The repository currently includes the following models.

**Models in published papers**

| Model     | Full name                                                                 | Paper                                              |
| --------- | ------------------------------------------------------------------------- | -------------------------------------------------- |
| NRMS      | Neural News Recommendation with Multi-Head Self-Attention                 | https://www.aclweb.org/anthology/D19-1671/         |
| NAML      | Neural News Recommendation with Attentive Multi-View Learning             | https://arxiv.org/abs/1907.05576                   |
| LSTUR     | Neural News Recommendation with Long- and Short-term User Representations | https://www.aclweb.org/anthology/P19-1033.pdf      |
| TANR      | Neural News Recommendation with Topic-Aware News Representation           | https://www.aclweb.org/anthology/P19-1110.pdf      |
| CAUM      | Candidate-Aware User Modeling												| https://dl.acm.org/doi/10.1145/3477495.3531778     |

**Experimental models**

| Model			| Description                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------- |
| CAUM			| Using different news information Only																 |
| CAUM-BERT		| CAUM + BERT in News Model													 |

## Get started

Basic setup.

```bash
git clone <your_repo>
cd NewsRecommendation
pip3 install -r requirements.txt
```

Download and preprocess the data for MIND dataset.

```bash
mkdir data && cd data
# Download GloVe pre-trained word embedding
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
sudo apt install unzip
unzip glove.840B.300d.zip -d glove
rm glove.840B.300d.zip

# Download MIND dataset
# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.

# Uncomment the following lines to use the MIND Large dataset (Note MIND Large test set doesn't have labels, see #11)
# wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip https://mind201910small.blob.core.windows.net/release/MINDlarge_test.zip
# unzip MINDlarge_train.zip -d train
# unzip MINDlarge_dev.zip -d val
# unzip MINDlarge_test.zip -d test
# rm MINDlarge_*.zip

# Uncomment the following lines to use the MIND Small dataset (Note MIND Small doesn't have a test set, so we just copy the validation set as test set :)
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_train.zip -d train
unzip MINDsmall_dev.zip -d val
cp -r val test # MIND Small has no test set :)
rm MINDsmall_*.zip

# Preprocess data into appropriate format
cd ..
python3 src/data_preprocess.py
# Remember you shoud modify `num_*` in `src/config.py` by the output of `src/data_preprocess.py`
```

Modify `src/config.py` to select target model. The configuration file is organized into general part (which is applied to all models) and model-specific part (that some models not have).

```bash
vim src/config.py
```

Run.

```bash
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

You can visualize metrics with TensorBoard.

```bash
tensorboard --logdir=runs

# or
tensorboard --logdir=runs/{model_name}
# for a specific model
```

> Tip: by adding `REMARK` environment variable, you can make the runs name in TensorBoard more meaningful. For example, `REMARK=num-filters-300-window-size-5 python3 src/train.py`.

## Results

| Model     | AUC | MRR | nDCG@5 | nDCG@10 | Remark |
| --------- | --- | --- | ------ | ------- | ------ |
| NRMS      |     |     |        |         |        |
| NAML      |     |     |        |         |        |
| LSTUR     |     |     |        |         |        |
| DKN       |     |     |        |         |        |
| Hi-Fi Ark |     |     |        |         |        |
| TANR      |     |     |        |         |        |

You can verify the results by simply downloading them and running `MODEL_NAME=XXXX python3 src/evaluate.py`.

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Original code: https://github.com/yusanshi/news-recommendation
- BERT implementation code: https://github.com/limhj159/NewsRecommendation
- CAUM code: https://github.com/andreeaiana/simplifying_nnr
