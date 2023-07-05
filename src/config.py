import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'Exp1'
# Currently included model
assert model_name in [
    'NRMS', 'NAML', 'LSTUR', 'DKN', 'HiFiArk', 'TANR', 'CAUM', 'Exp1'
]

class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_epochs = 2
    num_batches_show_loss = 200  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 4500
    batch_size = 128
    learning_rate = 0.0001
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    num_words_content = 100
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 2  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_users = 1 + 29120
    num_words = 1 + 271060
    num_categories = 1 + 148
    num_entities = 1 + 12957
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For multi-head self-attention
    num_attention_heads = 15


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title','category', 'subcategory', 'abstract'],
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']
    }
    # For CNN
    num_filters = 300
    window_size = 3
    long_short_term_method = 'con'
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}
    # For CNN
    num_filters = 50
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1

class CAUMConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract', 'content'],
        "record": []
    }
    batch_size = 32
    #learning_rate = 0.000005

    # For multi-head self-attention
    num_attention_heads = 20
    news_vector_dim = 400
    num_filters = 400
    #window_size = 3

    user_vector_dim = 400
    dense_att_hidden_dim1 = 400
    dense_att_hidden_dim2 = 256

class Exp1Config(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'content'],
        "record": []
    }

    batch_size = 32

    # For multi-head self-attention
    num_attention_heads = 20
    news_vector_dim = 400
    num_filters = 400

    user_vector_dim = 400
    dense_att_hidden_dim1 = 400
    dense_att_hidden_dim2 = 256

    bert_level = os.environ[
        'BERT_LEVEL'] if 'BERT_LEVEL' in os.environ else 'sentence'
    assert bert_level in ['word', 'sentence']

    fine_tune = False

    if fine_tune:
        for x in ['title', 'content']:
            if x in dataset_attributes['news']:
                dataset_attributes['news'].remove(x)
                dataset_attributes['news'].extend(
                    [f'{x}_bert', f'{x}_mask_bert'])