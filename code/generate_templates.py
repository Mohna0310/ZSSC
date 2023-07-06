# Script to generate templates
# Authors: Adithya Kulkarni, Mohna Chakraborty

import pickle
from transformers import BertTokenizer, BertForMaskedLM
import torch
import nltk
import pandas as pd
from nltk.corpus import wordnet


def parse_input_template(input_template, input_mapping):
    template = input_template.replace("<review>", "")
    template = template.replace(".", "")
    template = template.strip()
    mapping_keys = list(input_mapping.keys())
    template = template.replace("_", mapping_keys[0])
    return template


def positional_feature(parsed_input_template):
    return ["<review> . " + str(parsed_input_template) + " .", str(parsed_input_template) + " ." + " <review> ."]


def reasoning_feature(parsed_input_template):
    return ["<review> so " + str(parsed_input_template) + " .", str(parsed_input_template) + " because " + " <review> ."]


def paraphrase_feature(parsed_input_template, paraphrasing_tokens):
    templates = []
    template_split = parsed_input_template.split(" ")
    for i in range(0, len(template_split) - 1):
        pos_tag = nltk.pos_tag([template_split[i]])[0][1]
        print(pos_tag)
        for j in range(0, len(paraphrasing_tokens[i])):
            pos_tag1 = nltk.pos_tag([paraphrasing_tokens[i][j]])[0][1]
            if pos_tag1 == pos_tag:
                update_template = template_split.copy()
                update_template[i] = paraphrasing_tokens[i][j]
                templates.append(" ".join(update_template))
    return templates


def generate_templates(parsed_input_template, paraphrasing_tokens):
    paraphrase_templates = paraphrase_feature(parsed_input_template, paraphrasing_tokens)
    templates = []
    for i in range(0, len(paraphrase_templates)):
        positional_template = positional_feature(paraphrase_templates[i])
        reasoning_template = reasoning_feature(paraphrase_templates[i])
        for j in range(0, len(positional_template)):
            templates.append(positional_template[j])
        for j in range(0, len(reasoning_template)):
            templates.append(reasoning_template[j])
    return templates


def parse_mapping(input_mapping):
    keys = list(input_mapping.keys())
    mapping_tokens = []
    for i in range(0, len(keys)):
        mapping_tokens.append(input_mapping[keys[i]])
    return keys, mapping_tokens


if __name__ == "__main__":
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

    combined_sentence = train_sentence_list + test_sentence_list

    input_mapping = {'positive': 'great', 'negative': 'terrible'}
    task_details = ['sentence']
    input_template = "<review>. the sentence was _"
    parsed_input_template = parse_input_template(input_template, input_mapping)
    print(parsed_input_template)
    mapping_keys, mapping_tokens = parse_mapping(input_mapping)
    print(mapping_keys)
    print(mapping_tokens)

    # Either obtain synonym list from WordNet or define manually
    # synonyms_list = []
    # for i in range(0, len(mapping_tokens)):
    #     phrase = mapping_tokens[i]
    #     synonyms = []
    #     for syn in wordnet.synsets(phrase):
    #         for l in syn.lemmas():
    #             if l.name() not in synonyms:
    #                 synonyms.append(l.name())
    #     synonyms_list.append(synonyms)

    synonyms_list = [['outstanding', 'smashing', 'big', 'large'],
                     ['awful', 'horrendous', 'horrific', 'dreadful', 'direful', 'dire', 'unspeakable']]

    sample_train_review = []
    sample_train_label = []
    for i in range(0, len(combined_sentence)):
        review = str(combined_sentence[i]).lower()
        split_review = review.split(" ")
        for j in range(0, len(mapping_tokens)):
            temp = []
            temp_label = []
            if mapping_tokens[j] in split_review:
                temp.append(review)
                temp_label.append(0)
                if j == 0:
                    temp.append(str(review).replace(mapping_tokens[j], mapping_tokens[1]))
                    temp_label.append(1)
                    for k in range(0, len(synonyms_list[j])):
                        temp.append(str(review).replace(mapping_tokens[j], synonyms_list[j][k]))
                        temp_label.append(0)
                    for k in range(0, len(synonyms_list[1])):
                        temp.append(str(review).replace(mapping_tokens[j], synonyms_list[1][k]))
                        temp_label.append(1)
                elif j == 1:
                    temp.append(str(review).replace(mapping_tokens[j], mapping_tokens[0]))
                    temp_label.append(1)
                    for k in range(0, len(synonyms_list[j])):
                        temp.append(str(review).replace(mapping_tokens[j], synonyms_list[j][k]))
                        temp_label.append(0)
                    for k in range(0, len(synonyms_list[0])):
                        temp.append(str(review).replace(mapping_tokens[j], synonyms_list[1][k]))
                        temp_label.append(1)
            if len(temp) > 0 and len(temp_label) > 0:
                sample_train_review.append(temp)
                sample_train_label.append(temp_label)

    print(len(sample_train_review))
    print(len(sample_train_label))
    print(sample_train_review[0])
    print(sample_train_label[0])

    sample_review = ""
    for i in range(0, len(combined_sentence)):
        review = str(combined_sentence[i]).lower()
        split_review = review.split(" ")
        if mapping_tokens[0] in split_review:
            sample_review = review
            break

    template_split = parsed_input_template.split(" ")
    print(template_split)

    sample_templates = []

    for i in range(0, len(template_split) - 1):
        update_template = template_split.copy()
        update_template[i] = "[MASK]"
        sample_templates.append(str(" ".join(update_template)) + " because " + str(sample_review))

    print(sample_templates)

    paraphrasing_tokens = []

    for i in range(0, len(sample_templates)):
        input_review = sample_templates[i].strip()
        inputs = tokenizer(input_review, return_tensors='pt')
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        input_ids = inputs['input_ids'][0].tolist()
        # print(input_ids)
        index = input_ids.index(103)
        prediction = outputs.logits[0].tolist()[index]
        labels = sorted(range(len(prediction)), key=lambda i: prediction[i], reverse=True)[:30]
        res_list = [prediction[i] for i in labels]
        pred_tokens = tokenizer.convert_ids_to_tokens(labels)
        paraphrasing_tokens.append(pred_tokens)

    print(paraphrasing_tokens)

    templates = generate_templates(parsed_input_template, paraphrasing_tokens)
    print(templates)
    print(len(templates))

    dictionary = {'templates': templates, 'paraphrasing_tokens': paraphrasing_tokens,
                  'sample_train_review': sample_train_review, 'sample_train_label': sample_train_label}

    with open("../files/BERT_large_cr_paraphrasing_30_with_synonyms.pickle", 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

