import numpy as np
import nmslib

class search_files:
    
    def __init__(self, load_path = None, predictions = None):
        

        # Create new index on the predictions
        self.index = nmslib.init(method = 'hnsw', space = 'cosinesimil')
        self.index.addDataPointBatch(predictions)
        self.index.createIndex({'post': 2}, print_progress=True)

            
    def evaluate(self,paraphrase_embs,k=3):
        # Evaluate the paraphrase transcripts
        tp = 0
        
        # for each paraphrased transcript
        for i in range(len(paraphrase_embs)):
            r,d = self.index.knnQuery(paraphrase_embs[i][1:],k=k) # fetch k neighbors for it
            if int(paraphrase_embs[i][0]) in r:
                tp+=1 # if the corresponding audio file is in the neighbors increase the count
        return tp/len(paraphrase_embs) # fetched / total_length
        

preds = np.load('predictions.npy')
paraembs = np.load('paraphrase_embs.npy')

sf = search_files(predictions = preds)

for k in [1,3,5,10]: # number of neighbors
    print('\nk = '+str(k)+' acc. - '+ str(sf.evaluate(paraembs,k)))