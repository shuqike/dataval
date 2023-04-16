# dataval

> Machine learning is going real-time ... Latency matters, especially for user-facing applications. In 2009, Google’s experiments demonstrated that increasing web search latency 100 to 400 ms reduces the daily number of searches per user by 0.2% to 0.6%. In 2019, Booking.com found that an increase of 30% in latency cost about 0.5% in conversion rates — “a relevant cost for our business.”
>
> ---Chip Huyen

## Implementation plan

- [ ] Online supervised learning
- [ ] Online Truncated Monte Carlo

## Libraries

1. [River](https://github.com/online-ml/river/). River is a Python library for online machine learning/streaming data.

## Dataset pool

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
e

## References

1. [amiratag/DataShapley](https://github.com/amiratag/DataShapley). Official implementation of Data Shapley.
2. [MIT 6.883, Online Methods in Machine Learning: Theory and Applications, 2016](https://www.mit.edu/~rakhlin/6.883/). Outdated lecture on online ML.
3. [Awesome Online Machine Learning](https://github.com/online-ml/awesome-online-machine-learning)
4. [The correct way to evaluate online machine learning models, 2020](https://maxhalford.github.io/blog/online-learning-evaluation/)
5. [Real-time machine learning: challenges and solutions, 2022](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html)
