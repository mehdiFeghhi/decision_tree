class DecisionTree_Binery_Res_Gini:
    def __init__(self,list_of_feature_name,max_depth=100):
        self.list_of_feature_name = list_of_feature_name
        self.max_depth = max_depth
        self.first = True
        self.root = Node
    def _is_finished(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1):
            return True
        return False
    
    def _gini(self, y):
        
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1-p) for p in proportions if p > 0])
        return gini

    def _create_split(self, X, list_of_category):

        dic_of_variable = {}
        for item in list_of_category :
                idx = np.argwhere(X == item).flatten()
                dic_of_variable[item] = idx

        return dic_of_variable



    def _impurity_gini(self, X, y, list_of_category):
        dic_of_variable = self._create_split(X, list_of_category)
        main_gini = self._gini(y)
        print("main:")
        print(main_gini)
        n = len(y)
        gini_impurity = 0
        for key in  dic_of_variable:
            # if len(dic_of_variable[key]) != 0 :
              gini_impurity += ((len(dic_of_variable[key])/n) * self._gini(y[dic_of_variable[key]]))

        return main_gini - gini_impurity

    def _best_split(self, X, y, features):
        split_score = -1  
        split_feat = None
        list_of_category = None
        for feat in features:
            X_feat = X[:, feat]
            all_data_feat = self.data[:,feat]
            list_of_category = np.unique(all_data_feat)
            # for thresh in thresholds:
            score = self._impurity_gini(X_feat, y, list_of_category)
            # print(score)
            if score > split_score:
                print(score)
                split_score = score
                split_feat = feat
                list_of_category = list_of_category
        print(list_of_category)
        print(split_score)
        print(split_feat)
        return split_feat, list_of_category



    def tree_show(self):
      
        return self.root.show_node()

    def make_plot(self,name):

        return self.root.make_plot(name)

    
    def _find_perobabilty_of_pasetive(self,numpy_array,perobabilty_of_pasetive_of_father):

          
          unique_list, counts = np.unique(numpy_array, return_counts=True)

           
          if len(unique_list) == 1:

            if unique_list[0] == 1:

                return 1
            
            else:

                return 0

          if unique_list[0] == 1:
              number_pos = counts[0]
            
          else :
              number_pos = counts[1]

            
          if number_pos / (counts[0]+counts[1]) == 0.5:

              return perobabilty_of_pasetive_of_father

          else :

              return number_pos / (counts[0]+counts[1])


            






    def _build_tree(self, X, y,map_feature_number:list,perobabilty_of_pasetive_of_father,choice_from_father,depth=0):

        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
        
        if self.first == True:
            self.data = X
            self.first = False
            

        perobabilty_of_pasetive = self._find_perobabilty_of_pasetive(y,perobabilty_of_pasetive_of_father)
        # stopping criteria
        if self._is_finished(depth) or len(map_feature_number)==0:
            if len(y) == 0 :
                return Node(choice_from_father = choice_from_father ,value=perobabilty_of_pasetive_of_father)
            return Node(choice_from_father = choice_from_father , value=perobabilty_of_pasetive)

        # get best split
        best_feat, list_of_category = self._best_split(X, y, map_feature_number)
        map_feature_number_copy = map_feature_number.copy()
        map_feature_number_copy.remove(best_feat)
        # grow children recursively
        dic_of_variable = self._create_split(X[:, best_feat], list_of_category)
        
        list_of_childs = []
        
        print(dic_of_variable)

        for key in dic_of_variable:
            
            most_commen_child = None
            number_item_most_commen = -1
            helper = 0
            # print(dic_of_variable)
            print(dic_of_variable[key])
            if len(dic_of_variable[key]) != 0 :
                  child = self._build_tree(X[dic_of_variable[key],:],
                                                    y[dic_of_variable[key]],
                                                    map_feature_number_copy,
                                                    perobabilty_of_pasetive_of_father,
                                                    key,
                                                    depth + 1)
                  list_of_childs.append(child)
                  
                  helper = len(X[dic_of_variable[key],:])
                  # print(helper)
                  if helper > number_item_most_commen :
                      # print("hello")
                      most_commen_child = child

            else:
                  
                  list_of_childs.append(Node(feature=None, choice_from_father=key, list_of_childs = None, value=perobabilty_of_pasetive_of_father))
              



        return Node(self.list_of_feature_name[best_feat],best_feat,most_commen_child,choice_from_father,list_of_childs)
    
    def _traverse_tree(self, x, node):


        if node.is_leaf():
            return node.value
        print(node.childs)

        for child in node.childs:
          if x[node.feat_number] == child.choice_from_father:
            return self._traverse_tree(x,child)
        
        # print("hi")

        return self._traverse_tree(x,node.most_commen_child)


    def fit(self, X, y,map_feature_number):
        map_feature_number = map_feature_number

        firt_prob = self._find_perobabilty_of_pasetive(y,0.5)
        choice_from_father = "root"
        self.root = self._build_tree(X, y,map_feature_number,firt_prob,choice_from_father)

    def predict(self,X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def predict_by_thershold_out_put(self,X,thershold):

        predict = self.predict(X)
        res = []
        for i in predict:
          if i >= thershold:
            res.append(1)
          else:
            res.append(0)
        return np.array(res)
