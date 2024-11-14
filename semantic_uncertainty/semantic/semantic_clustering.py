from semantic_entropy import get_semantic_ids
from semantic_entropy import cluster_assignment_entropy

class SemanticClustering:
    def __init__(self,entailment_model) -> None:
        self.entailment_model=entailment_model

    def get_entropy(self,responses:list,example:str):
        '''return semantic entropy for one question
        responses: a list that contains different generations for 'example'
        example: one question in input dataset
        '''
        semantic_ids=get_semantic_ids(
                responses, model=self.entailment_model,
                strict_entailment=True, example=example)
        entropy=cluster_assignment_entropy(semantic_ids)
        return entropy


    
    
