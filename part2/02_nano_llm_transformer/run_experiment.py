"""Educational NumPy nano language model with causal self-attention. Author: Weihao Fu."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent; DATA=BASE/"data"; IMAGES=BASE/"images"; RESULTS=BASE/"results"; MODELS=BASE/"models"
TEXT="""data science turns observations into useful evidence. machine learning finds patterns in data. careful validation prevents misleading conclusions. clear visualizations help people understand results. reproducible code makes experiments trustworthy. customer segments reveal different behaviors. regression estimates numeric outcomes. classification predicts categories. clustering discovers groups without labels. association rules reveal products that occur together. """*12

def softmax(x):
    x=x-x.max(axis=-1,keepdims=True); e=np.exp(x); return e/e.sum(axis=-1,keepdims=True)

class NanoAttentionLM:
    def __init__(self,vocab,context=24,d=32,seed=42):
        self.vocab=vocab; self.stoi={c:i for i,c in enumerate(vocab)}; self.itos=dict(enumerate(vocab)); self.context=context; self.d=d
        r=np.random.default_rng(seed); scale=.18
        self.E=r.normal(0,scale,(len(vocab),d)); self.P=r.normal(0,scale,(context,d))
        self.Wq=r.normal(0,scale,(d,d)); self.Wk=r.normal(0,scale,(d,d)); self.Wv=r.normal(0,scale,(d,d)); self.Wo=r.normal(0,scale,(d,len(vocab)))
    def representation(self,ids):
        h=self.E[ids]+self.P[-len(ids):]; q=h@self.Wq; k=h@self.Wk; v=h@self.Wv
        scores=q@k.T/np.sqrt(self.d); mask=np.triu(np.ones_like(scores,dtype=bool),1); scores[mask]=-1e9
        attended=softmax(scores)@v; return (h+attended)[-1]
    def train(self,text,epochs=400,lr=.8):
        ids=np.array([self.stoi[c] for c in text]); X=[]; y=[]
        for i in range(self.context,len(ids)): X.append(self.representation(ids[i-self.context:i])); y.append(ids[i])
        X=np.asarray(X); y=np.asarray(y); losses=[]
        for epoch in range(epochs):
            probs=softmax(X@self.Wo); loss=-np.log(probs[np.arange(len(y)),y]+1e-12).mean(); losses.append(loss)
            grad=probs; grad[np.arange(len(y)),y]-=1; self.Wo-=lr*(X.T@grad/len(y)+1e-4*self.Wo)
        return losses
    def generate(self,prompt,length=180,seed=42):
        r=np.random.default_rng(seed); out=prompt.lower()
        for _ in range(length):
            ids=[self.stoi.get(c,self.stoi[" "]) for c in out[-self.context:]]; p=softmax(self.representation(ids)@self.Wo/.75)
            out+=self.itos[int(r.choice(len(self.vocab),p=p))]
        return out

def main():
    for d in (DATA,IMAGES,RESULTS,MODELS): d.mkdir(parents=True,exist_ok=True)
    (DATA/"training_corpus.txt").write_text(TEXT)
    vocab=sorted(set(TEXT)); model=NanoAttentionLM(vocab); losses=model.train(TEXT); sample=model.generate("data science ")
    pd.DataFrame({"epoch":range(1,len(losses)+1),"cross_entropy_loss":losses}).to_csv(RESULTS/"training_history.csv",index=False)
    (RESULTS/"generated_text.txt").write_text(sample+"\n")
    summary={"author":"Weihao Fu","architecture":"single-head causal self-attention with trained output projection","characters":len(TEXT),
             "vocabulary_size":len(vocab),"context_length":model.context,"embedding_dimension":model.d,"initial_loss":round(losses[0],4),"final_loss":round(losses[-1],4)}
    (RESULTS/"experiment_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez(MODELS/"nano_attention_model.npz",E=model.E,P=model.P,Wq=model.Wq,Wk=model.Wk,Wv=model.Wv,Wo=model.Wo,vocab=np.array(vocab))
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(range(1,len(losses)+1),losses,color="#4e79a7",lw=2); ax.set(xlabel="Epoch",ylabel="Cross-Entropy Loss",title="Nano Attention Language Model Training"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(IMAGES/"training_loss.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    print("Nano language-model experiment completed."); print(json.dumps(summary,indent=2)); print("\nSample:\n",sample[:260])
if __name__=="__main__": main()
