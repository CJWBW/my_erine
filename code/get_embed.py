import torch
from pathlib import Path
import pickle

KG_EMBED_PATH = str(Path(__file__).parent / "kg_embed")


def main():
    pass

    '''Altogether will out of memory so we split to two parts. we get the tensor separately and combine them before
     converting to one embedding'''

    # vecs = [[0] * 100]
    # with open(KG_EMBED_PATH + "/entity2vec.vec", 'r') as fin:
    #     a = 1
    #     for line in fin:
    #         if a < 3000000:
    #             vec = line.strip().split('\t')
    #             vec = [float(x) for x in vec]
    #             vecs.append(vec)
    #         a += 1
    # embed = torch.FloatTensor(vecs)
    # # embed = torch.nn.Embedding.from_pretrained(embed)
    # with open('tensor_embed1.txt', 'wb') as f:
    #     pickle.dump(embed, f)

    # vecs = [[0] * 100]
    # with open(KG_EMBED_PATH + "/entity2vec.vec", 'r') as fin:
    #     a = 1
    #     for line in fin:
    #         if a >= 3000000:
    #             vec = line.strip().split('\t')
    #             vec = [float(x) for x in vec]
    #             vecs.append(vec)
    #         a += 1
    # embed = torch.FloatTensor(vecs)
    # # embed = torch.nn.Embedding.from_pretrained(embed)
    # with open('tensor_embed2.txt', 'wb') as f:
    #     pickle.dump(embed, f)


    with open('tensor_embed1.txt', 'rb') as f:
        tensor_e1 = pickle.load(f)

    with open('tensor_embed2.txt', 'rb') as f:
        tensor_e2 = pickle.load(f)

    '''now we concatenate the two tensors along the 0-dimension'''
    embed = torch.cat((tensor_e1, tensor_e2), 0)
    embed = torch.nn.Embedding.from_pretrained(embed)
    with open('embed.txt', 'wb') as f:
        pickle.dump(embed, f)

    with open('embed.txt', 'rb') as f:
        embed = pickle.load(f)

    a = 1





if __name__ == "__main__":
    main()
