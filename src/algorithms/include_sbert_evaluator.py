import logging
import os
import csv
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
import numpy as np
from typing import List
from sentence_transformers import InputExample
from sentence_transformers.evaluation import BinaryClassificationEvaluator


logger = logging.getLogger(__name__)

class CustomBinaryClassificationEvaluator(BinaryClassificationEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    """

    def __init__(self, sentences1: List[str], sentences2: List[str], labels: List[int], name: str = '', batch_size: int = 32, show_progress_bar: bool = False, write_csv: bool = True, targets={'recall': [1., 0.9, 0.8, 0.7, 0.6, 0.5]}):
        # super().__init__(sentences1, sentences2, labels, name, name, batch_size, show_progress_bar, write_csv)
        self.targets = targets
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "binary_classification_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "cossim_accuracy", "cossim_accuracy_threshold", "cossim_f1", "cossim_precision", "cossim_recall", "cossim_f1_threshold", "cossim_ap",
                            "manhattan_accuracy", "manhattan_accuracy_threshold", "manhattan_f1", "manhattan_precision", "manhattan_recall", "manhattan_f1_threshold", "manhattan_ap",
                            "euclidean_accuracy", "euclidean_accuracy_threshold", "euclidean_f1", "euclidean_precision", "euclidean_recall", "euclidean_f1_threshold", "euclidean_ap"
                           ]
        self.csv_extra_headers = []
        # for d in ['cossim', 'manhattan', 'euclidean', 'dot']:
        for d in ['cossim', 'manhattan', 'euclidean']:
            for m in ['f1', 'precision', 'recall', 'threshold']:
                for t in self.targets['recall']:
                    self.csv_extra_headers.append(f'{d}_{m}_{t}')
        self.csv_headers.extend(self.csv_extra_headers)

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Binary Accuracy Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)


        #Main score is the max of Average Precision (AP)
        main_score = max(scores[short_name]['ap'] for short_name in scores)

        file_output_data = [epoch, steps]

        for header_name in self.csv_headers:
            if '_' in header_name:
                sim_fct, metric = header_name.split("_", maxsplit=1)
                file_output_data.append(scores[sim_fct][metric])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score


    def compute_metrices(self, model):
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
        embeddings1 = [emb_dict[sent] for sent in self.sentences1]
        embeddings2 = [emb_dict[sent] for sent in self.sentences2]

        i = 0
        cosine_scores = []
        manhattan_distances = []
        euclidean_distances = []
        while i < len(self.sentences1):
            if i + self.batch_size >= len(self.sentences1):
                embed1, embed2 = embeddings1[i:i+self.batch_size], embeddings2[i:]
            else:
                embed1, embed2 = embeddings1[i:i+self.batch_size], embeddings2[i:i+self.batch_size]
            i += self.batch_size
            cosine_scores.extend(1 - paired_cosine_distances(embed1, embed2))
            manhattan_distances.extend(paired_manhattan_distances(embed1, embed2))
            euclidean_distances.extend(paired_euclidean_distances(embed1, embed2))

        cosine_scores = np.asarray(cosine_scores)
        manhattan_distances = np.asarray(manhattan_distances)
        euclidean_distances = np.asarray(euclidean_distances)
        # embeddings1_np = np.asarray(embeddings1)
        # embeddings2_np = np.asarray(embeddings2)
        # dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]


        labels = np.asarray(self.labels)
        output_scores = {}
        # for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True], ['manhattan', 'Manhattan-Distance', manhattan_distances, False], ['euclidean', 'Euclidean-Distance', euclidean_distances, False], ['dot', 'Dot-Product', dot_scores, True]]:
        for short_name, name, scores, reverse in [['cossim', 'Cosine-Similarity', cosine_scores, True], ['manhattan', 'Manhattan-Distance', manhattan_distances, False], ['euclidean', 'Euclidean-Distance', euclidean_distances, False]]:
            acc, acc_threshold = self.find_best_acc_and_threshold(scores, labels, reverse)
            ap = average_precision_score(labels, scores * (1 if reverse else -1))
            f1, precision, recall, f1_threshold = self.find_best_f1_and_threshold(scores, labels, reverse)
            tar_metrics = {}
            for tar_recall in self.targets['recall']:
                f1_tar, precision_tar, recall_tar, thresh_tar = self.find_threshold_with_target(scores, labels, tar_recall, reverse)
                tar_metrics[f'f1_{tar_recall}'] = f1_tar
                tar_metrics[f'precision_{tar_recall}'] = precision_tar
                tar_metrics[f'recall_{tar_recall}'] = recall_tar
                tar_metrics[f'threshold_{tar_recall}'] = thresh_tar

            logger.info("Accuracy with {}:           {:.2f}\t(Threshold: {:.4f})".format(name, acc * 100, acc_threshold))
            logger.info("F1 with {}:                 {:.2f}\t(Threshold: {:.4f})".format(name, f1 * 100, f1_threshold))
            logger.info("Precision with {}:          {:.2f}".format(name, precision * 100))
            logger.info("Recall with {}:             {:.2f}".format(name, recall * 100))
            logger.info("Average Precision with {}:  {:.2f}\n".format(name, ap * 100))

            output_scores[short_name] = {
                'accuracy' : acc,
                'accuracy_threshold': acc_threshold,
                'f1': f1,
                'f1_threshold': f1_threshold,
                'precision': precision,
                'recall': recall,
                'ap': ap
            }
            output_scores[short_name].update(tar_metrics)


        return output_scores



    @staticmethod
    def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)
        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        max_acc = 0
        best_threshold = -1

        positive_so_far = 0
        remaining_negatives = sum(labels == 0)

        for i in range(len(rows)-1):
            score, label = rows[i]
            if label == 1:
                positive_so_far += 1
            else:
                remaining_negatives -= 1

            acc = (positive_so_far + remaining_negatives) / len(labels)
            if acc > max_acc:
                max_acc = acc
                best_threshold = (rows[i][0] + rows[i+1][0]) / 2

        return max_acc, best_threshold

    @staticmethod
    def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        rows = list(zip(scores, labels))

        rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

        best_f1 = best_precision = best_recall = 0
        threshold = 0
        nextract = 0
        ncorrect = 0
        total_num_duplicates = sum(labels)

        for i in range(len(rows)-1):
            score, label = rows[i]
            nextract += 1

            if label == 1:
                ncorrect += 1

            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / total_num_duplicates
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    threshold = (rows[i][0] + rows[i + 1][0]) / 2

        return best_f1, best_precision, best_recall, threshold
    
    @staticmethod
    def find_threshold_with_target(scores, labels, tar, high_score_more_similar):
        assert len(scores) == len(labels)

        scores = np.asarray(scores)
        labels = np.asarray(labels)
        # def find_thresh(y_true, score, tar={'recall': 0.8}):
        if len(scores.shape) > 1:
            scores = scores[:, 1]
        
        ps, rs, ths = precision_recall_curve(labels, scores)
        idx = np.argmin(np.abs(rs - tar))
        thresh = ths[idx]
        if high_score_more_similar:
            preds = scores > thresh
        else:
            preds = scores < thresh
        f1 = 2 * (ps[idx] * rs[idx]) / (ps[idx] + rs[idx])
        
        return f1, ps[idx], rs[idx], thresh