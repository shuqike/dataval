# dataval

## Motivation

Why do we need to consider an online setting for data valuation?

> Machine learning is going real-time ... Latency matters, especially for user-facing applications. In 2009, Google’s experiments demonstrated that increasing web search latency 100 to 400 ms reduces the daily number of searches per user by 0.2% to 0.6%. In 2019, Booking.com found that an increase of 30% in latency cost about 0.5% in conversion rates — “a relevant cost for our business.”
>
> ---Chip Huyen

Why do we need to consider vector database? Search engine provides a nice online setting. There are some use cases:

> - Semantic text search: Convert text data into vector embeddings using an NLP transformer such as a sentence embedding model, then index and search through those vectors using Pinecone.
> - Generative question-answering: Retrieve relevant contexts to queries from Pinecone and pass these to a generative model like OpenAI to generate an answer backed by real data sources.
> - Hybrid search: Perform semantic and keyword search over your data in one query and combine the results for more relevant results.
> - Image similarity search: Transform image data into vector embeddings and build an index with Pinecone. Then convert query images into vectors and retrieve similar images.
> - Product recommendations: Generate product recommendations for ecommerce based on vectors representing users.
>
> ---Pinecone Document

## Implementation plan

- [ ] Online supervised learning/continual learning algorithms
- [ ] Online Truncated Monte Carlo
- [ ] Mean field inference learning
- [ ] Connect the data valuation to vector database

## Comparison

We compare the time and memory consumption of different setting pairs:

- W/ and w/o data removal
- Stateless/stateful training: the model is trained from scratch or the model continues training on new data (fine-tuning)
- Single/MultiModal

## Libraries

1. [River](https://github.com/online-ml/river/). River is a Python library for online machine learning/streaming data.
2. [Flink](https://flink.apache.org/). Apache Flink is a framework and distributed processing engine for stateful computations over unbounded and bounded data streams.
3. [BARS Benchmark](https://openbenchmark.github.io/BARS/index.html#). BARS is a large, open benchmark for recommender systems.
4. [DeepCTR](https://github.com/shenweichen/DeepCTR). DeepCTR is a easy-to-use, modular and extendible package of deep-learning based CTR(Clickthrough rate prediction) models along with lots of core components layers which can be used to easily build custom models.

## Dataset pool

### Recommendation system

| Dataset | Timestamp | Timespan |
| ------- | --------- | -------- |
| [MIND: MIcrosoft News Dataset](https://msnews.github.io/) | NaN | 6 weeks |

### Networks

| Dataset | Timestamp | Timespan |
| ------- | --------- | -------- |
| [Social Network: MOOC User Action Dataset](https://snap.stanford.edu/data/act-mooc.html) | seconds | NaN |
| [Dynamic Face-to-Face Interaction Networks](https://snap.stanford.edu/data/comm-f2f-Resistance.html) | 1/3second | 142,005 seconds |
| [CollegeMsg temporal network](https://snap.stanford.edu/data/CollegeMsg.html) | NaN  | 193 days |
| [Super User temporal network](https://snap.stanford.edu/data/sx-superuser.html) | NaN | 2773 days |
| [Stack Overflow temporal network](https://snap.stanford.edu/data/sx-stackoverflow.html) | NaN | 2774 days |

### Time series

| Dataset | Timestamp | Timespan |
| ------- | --------- | -------- |
| [Individual household electric power consumption](https://archive-beta.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) | minutes | 4 years |
| [News Popularity in Multiple Social Media Platforms](https://archive-beta.ics.uci.edu/dataset/432/news+popularity+in+multiple+social+media+platforms) | minutes | 9 months |

### Rare event/threat detection

| Dataset | Timestamp | Timespan |
| ------- | --------- | -------- |
| [CSE-CIC-IDS2018 on AWS](https://www.unb.ca/cic/datasets/ids-2018.html) | NaN | 1-2days |

## References

1. [amiratag/DataShapley](https://github.com/amiratag/DataShapley). Official implementation of Data Shapley.
2. [MIT 6.883, Online Methods in Machine Learning: Theory and Applications, 2016](https://www.mit.edu/~rakhlin/6.883/). Outdated lecture on online ML.
3. [Awesome Online Machine Learning](https://github.com/online-ml/awesome-online-machine-learning)
4. [The correct way to evaluate online machine learning models, 2020](https://maxhalford.github.io/blog/online-learning-evaluation/)
5. [Real-time machine learning: challenges and solutions, 2022](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html)
6. [Threat Detection Using Pinecone](https://docs.pinecone.io/docs/it-threat-detection)
