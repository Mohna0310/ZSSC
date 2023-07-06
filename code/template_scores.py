# Script to validate the generated templates
# Authors: Adithya Kulkarni, Mohna Chakraborty

import pickle
from transformers import BertTokenizer, BertForMaskedLM
import torch
import numpy as np


def map_labels(mapping_keys):
    dictionary = {}
    for i in range(0, len(mapping_keys)):
        dictionary[mapping_keys[i]] = i
    return dictionary


def convert_labels_to_id(labels, dictionary):
    output = []
    for i in range(0, len(labels)):
        output.append(dictionary[labels[i]])
    return output


def parse_mapping(input_mapping):
    keys = list(input_mapping.keys())
    mapping_tokens = []
    for i in range(0, len(keys)):
        mapping_tokens.append(input_mapping[keys[i]])
    return keys, mapping_tokens


def format_review(review, template):
    review = review.replace(".", "")
    updated_template = template.replace("<review>", review)
    updated_template = updated_template.replace("  ", " ")
    updated_template = str(updated_template).lower()
    updated_template = updated_template.replace("positive", "[MASK]")
    return updated_template


def get_mapping_token_id(mapping_tokens, tokenizer):
    inputs = tokenizer(" ".join(mapping_tokens), return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    return input_ids[1:-1]


def get_prediction_label(prediction_probability):
    maximum = max(prediction_probability)
    return prediction_probability.index(maximum)


def get_prediction_probability(mapping_token_id, prediction):
    scores = []
    sum = 0
    for i in range(0, len(mapping_token_id)):
        scores.append(prediction[mapping_token_id[i]])
        sum = sum + np.exp(prediction[mapping_token_id[i]])

    prediction_probability = []
    for i in range(0, len(scores)):
        prediction_probability.append(np.exp(scores[i])/sum)

    return prediction_probability, get_prediction_label(prediction_probability)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased')

    with open("../files/BERT_large_cr_paraphrasing_30_with_synonyms.pickle", 'rb') as handle:
        dictionary = pickle.load(handle)

    templates = dictionary['templates']
    sample_train_review = dictionary['sample_train_review']
    sample_train_label = dictionary['sample_train_label']
    input_mapping = {'positive': 'great', 'negative': 'terrible'}
    mapping_keys, mapping_tokens = parse_mapping(input_mapping)
    mapping_token_id = get_mapping_token_id(mapping_tokens, tokenizer)
    print(mapping_token_id)

    template_correct_count = []

    final_template_prediction = []
    final_template_prediction_prob = []

    for i in range(0, len(templates)):
        count = 0
        inter_template_prediction = []
        inter_template_prediction_prob = []
        for j in range(0, len(sample_train_review)):
            template_prediction = []
            template_prediction_prob = []
            for k in range(0, len(sample_train_review[j])):
                sample_review = format_review(sample_train_review[j][k], templates[i])
                # print(sample_review)
                inputs = tokenizer(sample_review, return_tensors='pt')
                input_ids = inputs['input_ids'][0].tolist()
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
                index = input_ids.index(103)
                prediction = outputs.logits[0].tolist()[index]
                prediction_probability, predict_labels = get_prediction_probability(mapping_token_id, prediction)
                template_prediction.append(predict_labels)
                template_prediction_prob.append(prediction_probability)

            zero = template_prediction[0]
            if zero == 0:
                non_zero = 1
            else:
                non_zero = 0
            for k in range(0, len(template_prediction)):
                if sample_train_label[j][k] == 0:
                    if template_prediction[k] == zero:
                        count = count + 1
                elif sample_train_label[j][k] == 1:
                    if template_prediction[k] == non_zero:
                        count = count + 1

            inter_template_prediction.append(template_prediction)
            inter_template_prediction_prob.append(template_prediction_prob)
        template_correct_count.append(count)
        final_template_prediction.append(inter_template_prediction)
        final_template_prediction_prob.append(inter_template_prediction_prob)

    print(template_correct_count)
    print(max(template_correct_count))
    print(min(template_correct_count))

    dictionary['label_mapping'] = map_labels(mapping_keys)
    dictionary['template_prediction'] = final_template_prediction
    dictionary['template_prediction_prob'] = final_template_prediction_prob
    dictionary['accuracy_results'] = template_correct_count

    with open("../files/BERT_large_cr_paraphrasing_30_with_synonyms_template_scores.pickle", 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
