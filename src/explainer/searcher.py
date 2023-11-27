class Searcher:
    def __init__(
        self,
        variables: list
    ):
        '''
            This class implements the search strategy for the explainer.

            More details are provided in our paper on this.

            *Required parameters*
            - variables: list of variables to search over
        '''
        self.current_node = tuple()
        self.variables = variables
        self.complete_nodes = []
        self.search_queue = []

    def get_next_evaluation_set(
        self,
    ):
        '''
            This method returns the next set of variables to be evaluated by the explainer.

            *Required parameters*
            - None

            *Optional parameters*
            - None

            *Returns*
            - list of variables to be evaluated
        '''
        possible_next_steps = [
            (*self.current_node, variable) for variable in self.variables if variable not in list(self.current_node)
        ]
        if len(possible_next_steps)==0:
            self.select_next_node({})
            return self.get_next_evaluation_set()
        else:
            return possible_next_steps
    
    def select_next_node(
        self,
        results_dictionary: dict
    ):
        '''
            This method selects the next node to be worked down from.
            
            *Required parameters*
            - results_dictionary: dictionary of results from the previous evaluation

            *Optional parameters*
            - None

            *Returns*
            - tuple of variables to be evaluated
        '''
        sorted_node_choices = sorted(results_dictionary.keys(), key=results_dictionary.get, reverse=True)

        for (arg, node) in enumerate(sorted_node_choices):
            if node not in self.complete_nodes:
                self.current_node = node
                self.search_queue = sorted_node_choices[arg+1:] + self.search_queue
                return self.current_node

        self.complete_nodes.append(self.current_node)
        if len(self.search_queue) == 0:
            raise ValueError("All nodes in this graph have been searched")
        self.current_node = self.search_queue.pop(0)
        return self.current_node
