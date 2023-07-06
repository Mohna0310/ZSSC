# Script to evaluate the templates
# Authors: Adithya Kulkarni, Mohna Chakraborty

import pickle
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np
import pandas as pd


def parse_mapping(input_mapping):
    keys = list(input_mapping.keys())
    mapping_tokens = []
    for i in range(0, len(keys)):
        mapping_tokens.append(input_mapping[keys[i]])
    return keys, mapping_tokens


def convert_labels_to_id(labels, dictionary):
    output = []
    for i in range(0, len(labels)):
        output.append(dictionary[labels[i]])
    return output


def map_labels(mapping_keys):
    dictionary = {}
    for i in range(0, len(mapping_keys)):
        dictionary[mapping_keys[i]] = i
    return dictionary


def format_review(review, template):
    review = review.replace(".", "")
    updated_template = template.replace("<review>", review)
    updated_template = updated_template.replace("  ", " ")
    updated_template = str(updated_template).lower()
    updated_template = updated_template.replace("positive", "[MASK]")
    return updated_template


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


def get_mapping_token_id(mapping_tokens, tokenizer):
    inputs = tokenizer(" ".join(mapping_tokens), return_tensors='pt')
    input_ids = inputs['input_ids'][0].tolist()
    return input_ids[1:-1]


def evaluate_data(reviews, template):
    template_prediction = []
    template_prediction_prob = []
    for j in range(0, len(reviews)):
        sample_review = format_review(reviews[j].lower(), template)
        inputs = tokenizer(sample_review, return_tensors='pt')
        input_ids = inputs['input_ids'][0].tolist()
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        index = input_ids.index(103)
        prediction = outputs.logits[0].tolist()[index]
        prediction_probability, predict_labels = get_prediction_probability(mapping_token_id, prediction)
        template_prediction.append(predict_labels)
        template_prediction_prob.append(prediction_probability)
    return template_prediction, template_prediction_prob


def mv(t_prediction):
    positive_count = 0
    negative_count = 0
    for i in range(0, len(t_prediction)):
        if t_prediction[i] == 0:
            negative_count = negative_count + 1
        else:
            positive_count = positive_count + 1
    output = [negative_count, positive_count]
    maximum = max(output)
    max_index = output.index(maximum)
    return max_index


def majority_voting(prediction):
    aggregated_prediction = []
    for i in range(0, len(prediction[0])):
        t_predictions = []
        for j in range(0, len(prediction)):
            t_predictions.append(prediction[j][i])
        aggregated_prediction.append(mv(t_predictions))
    return aggregated_prediction


def evaluate_top_k_templates(reviews, sorted_templates, k):
    chosen_templates = sorted_templates[:k]
    template_prediction = []
    template_prediction_prob = []
    for i in range(0, len(chosen_templates)):
        suggested_prediction, suggested_prediction_prob = evaluate_data(reviews, chosen_templates[i])
        template_prediction.append(suggested_prediction)
        template_prediction_prob.append(suggested_prediction_prob)

    return majority_voting(template_prediction), template_prediction, template_prediction_prob


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForMaskedLM.from_pretrained('bert-large-uncased')
    train_data = pd.read_csv("../dataset/original/cr/train.csv")
    columns_list = list(train_data.columns)
    train_label_list = list(train_data[columns_list[0]])
    train_sentence_list = list(train_data[columns_list[1]])

    data = pd.read_csv("../dataset/original/cr/test.csv")
    columns_list = list(data.columns)
    test_label_list = list(data[columns_list[0]])
    test_sentence_list = list(data[columns_list[1]])

    with open("../files/BERT_large_cr_paraphrasing_30_with_synonyms_template_scores.pickle", 'rb') as handle:
        template_dictionary_eval = pickle.load(handle)

    with open("../files/BERT_large_cr_paraphrasing_30_with_synonyms.pickle", 'rb') as handle:
        dictionary = pickle.load(handle)

    accuracy_results = template_dictionary_eval['accuracy_results']
    templates = dictionary['templates']
    sample_train_review = dictionary['sample_train_review']
    sample_train_label = dictionary['sample_train_label']
    input_mapping = {'positive': 'great', 'negative': 'terrible'}
    mapping_keys, mapping_tokens = parse_mapping(input_mapping)
    mapping_token_id = get_mapping_token_id(mapping_tokens, tokenizer)

    sorted_accuracy, sorted_templates = zip(*sorted(zip(accuracy_results, templates), reverse=True))
    print(sorted_templates[:5])
    print(sorted_accuracy[:5])
    print(sample_train_review[0])

    train_gt_labels = train_label_list
    test_gt_labels = test_label_list

    print(sorted_templates[0])

    train_prompt_prediction, train_prompt_prediction_prob = evaluate_data(train_sentence_list, sorted_templates[0])
    test_prompt_prediction, test_prompt_prediction_prob = evaluate_data(test_sentence_list, sorted_templates[0])

    print("Prompt 1 Performance")
    print("Train set")
    print(accuracy_score(train_gt_labels, train_prompt_prediction))
    print(classification_report(train_gt_labels, train_prompt_prediction))

    print("Test set")
    print(accuracy_score(test_gt_labels, test_prompt_prediction))
    print(classification_report(test_gt_labels, test_prompt_prediction))

    save_results_dict = {}
    save_results_dict['train_reviews'] = train_sentence_list
    save_results_dict['test_reviews'] = test_sentence_list
    save_results_dict['train_gt_labels'] = train_gt_labels
    save_results_dict['test_gt_labels'] = test_gt_labels

    save_results_dict['train_prompt_prediction'] = train_prompt_prediction
    save_results_dict['train_prompt_prediction_prob'] = train_prompt_prediction_prob
    save_results_dict['test_prompt_prediction'] = test_prompt_prediction
    save_results_dict['test_prompt_prediction_prob'] = test_prompt_prediction_prob

    with open("../files/BERT_large_cr_evaluation.pickle", 'wb') as handle:
        pickle.dump(save_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
