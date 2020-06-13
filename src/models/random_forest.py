import syssys.path.append('src')import loggingimport pandas as pdimport numpy as npfrom collections import Counterfrom utils import get_input_var_pos,sub_sample_and_featureslogger = logging.getLogger(__name__)class RandomForest():        def __init__(self,decision_tree_type,n_trees,trained_tree=None):        self.decision_tree_type = decision_tree_type        self.n_trees = n_trees        self.trained_tree = trained_tree    def build_forest(self,df,sample_ratio,n_selected_features):                random_forest = []        labels = df.iloc[:,-1]                for _ in range(0,self.n_trees):            print("::::::::::::::TREE N° {}::::::::::::::".format(_))            #sample with replacement            input_data_tree = sub_sample_and_features(df,labels,n_selected_features,sample_ratio)                        header = input_data_tree.columns.values.tolist()            logger.info(header)            random_forest.append([header,self.decision_tree_type.build_tree(input_data_tree,header)])                return RandomForest(self.decision_tree_type, self.n_trees, random_forest)            def classify_forest(self,data_variables,data):                tree_predictions = Counter()                for j in self.trained_tree:                   tree = j[1]            selected_var = data[get_input_var_pos(data_variables,j[0])]                      pred = tree.classify(selected_var)            for key,value in pred.items():                tree_predictions[key] += value                final_prediction = tree_predictions.most_common()[0]                logger.info("Classificato come {} con {} voti".format(final_prediction[0],final_prediction[1]))              return final_prediction[0]                    def get_model_accuracy(self,data_variables,validate_data):        #TO-DO        labels = validate_data.iloc[:,-1].values        print(labels)                predictions = []        for i,row in enumerate(validate_data.values):                        pred = self.classify_forest(data_variables,row)            predictions.append(pred)            print("Y_pred= {}; Y= {}".format(pred,labels[i]))                predictions = np.array(predictions)                if len(labels) == len(predictions):                        if isinstance(predictions[0], str) and isinstance(labels[0], str):                                acc = [1 if x == y else 0  for x, y in zip(labels, predictions)]                acc_tot = sum(acc)/len(labels) * 100                print("Accuratezza modello del {}".format(acc_tot))                            if isinstance(predictions[0], int) and isinstance(labels[0], int):                                ESS = np.sum((predictions - labels.mean()) ** 2)                TSS = np.sum((labels - labels.mean())** 2)                print("Accuratezza modello del {}".format(ESS/TSS))        return predictions       