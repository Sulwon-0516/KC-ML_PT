# Show and Tell: A Neural Image Caption Generator

# Abstract+논문의 소개

연구의 당위성 : image를 설명하는 desscription을 자동으로 생성하는 것은 CV와 NLP를 잇는 근본적인 문제

연구의 방향성 : 

Model : CV의 CNN + NLP의 machine translation에 쓰이는 RNN(LSTM)을 붙여보자. 

Train : Target description과 생성된 문장의 likelihood를 maximize하자.

연구의 성과 : SBU, Flickr30k, Pascal dataset에 대해서 BLEU-1 score로 SOTA를 달성. COCO dataset에서 BLEU-4 27.7로 SOTA를 달성.

# Introduction

<연구의 배경>

CV에서 주로 연구하는 object recognition이나 image classification에 한발 더 나아가, object를 감지하고 그 결과를 natural language로 변환해야 한다는 점에서 challenging task이다.

기존의 연구의 경우 sub-problem들로 나눈 것을 합치는 방향으로 진행되었지만, 이 논문에선 end to end로 구현을 했다.

- $input\ image\ I$ 에 대해서 $p(S|I)$를 maximize하는 방식으로 구현했음. 이때 문장 S는 dictionary안에 들어있는 단어들의 배열로 구현

<RNN을 섞을 생각을 한 이유>

아래의 논문들과 같이, 2014년도의 Machine translation의 SOTA는 대부분 RNN을 통해서 이뤄져 있음. 

- D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate.
arXiv:1409.0473, 2014.
- K. Cho, B. van Merrienboer, C. Gulcehre, F. Bougares,
H. Schwenk, and Y. Bengio. Learning phrase representations
using RNN encoder-decoder for statistical machine translation. In EMNLP, 2014.
- I. Sutskever, O. Vinyals, and Q. V. Le. Sequence to sequence
learning with neural networks. In NIPS, 2014.

이들은 encoder RNN이 source sentence를 읽고서 fixed-length vector로 만든뒤, decoder RNN의 initial hidden state로 넣어주는 방식을 취한다.

<제시한 Model의 이론적 배경>

위의 machine translation에서 입력 신호가 seq of words에서 image로 바뀐 경우를 푸는 것이니 encoder RNN 대신 CNN을 쓰자. 

- CNN을 통해서 feature를 뽑아낼 수 있고, fixed-length vector에 이 feature들을 embedding할 수 있으니까.
- Image classification으로 pretratin을 한뒤 last-hidden layer를 가져오자.

우리가 제시하는 모델 이름을 Neural Image Caption (NIC)라고 정의하겠다.

# Related Work

<기존에 제시된 연구들의 단점>

CV에서 자연어로 description을 만드는 연구는 주로 video에 대해서 있었음 but hand-designed + limited domain

- D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate.
arXiv:1409.0473, 2014.
- 위의 논문은 2012년에 나온 논문으로 자동으로 syntatic tree를 만들어주는 generator를 만듦.

    당시 SOTA.

다른 논문에선 template을 이용해 문자로 바꾸는 연구들이 존재했음. 

- 하지만, hand-designed + 융통성이 부족했음
- 또한 이미지를 있는 그대로 묘사 (ITW) (detection된 결과 + relation의 조합 느낌)

Image와 문자들을 하나의 공간에 co-embedding하고서 그 거리를 통해 description을 얻으려고 시도했음.

- cropped Image + substring 으로도 시도.
- 하지만, unseen composition of object에 대해선 묘사를 못하는 단점.
- 또한 생성 결과의 evaluation을 제시하지 않음.

<이 연구의 차별성>

본 연구에서 사용된 모델들의 배경

BN + LSTM

- S. Ioffe and C. Szegedy. Batch normalization: Accelerating
deep network training by reducing internal covariate shift.
- S. Hochreiter and J. Schmidhuber. Long short-term memory.
Neural Computation, 9(8), 1997.

유사 연구 : 

- Kiros et al : NN을 feedfoward로만 사용
- Mao et al : 가장 유사한 연구

    아래와 같이 m-RNN이라는 모델을 만들었는데, CNN 끝에 나온 image feature를 initial hidden state가 아닌, 모든 cycle마다 넣어준 차이가 보인다.  

    그게 아니어도 model의 형태가 다른데, 결과적으로 더 우수한 성과를 거둔 본 논문이 더 대중의 선택을 받은게 아닐까 생각한다.

    ![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled.png)

- Kiros et al : joint multimodeal embedding space를 CNN+LSTM으로 구현

    이 연구는 LSTM과 CNN을 병렬적으로 구성해 multimodal space에 embedding했음.

    ![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%201.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%201.png)

# Model

아래의 형태로 구성되어 있다.

RNN에 다양한 길이의 입력을 넣고서 fixed dimensional vector로 feature이 표현되니 유리함.

주어진 입력 이미지 I에 대해서 S와의 likelihood를 maximize하는 형태로 모델을 학습시킨다.

S0~St-1까지의 입력(correct transcription)과 입력 이미지는 t=t의 hidden state의 fixed length vector로 표현이 된다.

- 전이학습

     ImageNet에 대해서 학습시킨 것을 사용하는 당위성을 보여줌. (전이학습의 배경)

    "Our visual results demonstrate the generality and semantic knowledge implicit in
    these features, showing that the features tend to cluster images into interesting semantic categories on which the network was never explicitly trained"

    DeCAF: A Deep Convolutional Activation Feature
    for Generic Visual Recognition

사용된 Model의 모습.

![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%202.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%202.png)

![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%203.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%203.png)

1. LSTM 기반 문장 생성기

    LSTM은 vanishing, exploding gradients issue를 피할 수 있는 RNN model이다.

    우상단의 그림에서 **C** 가 memory cell로 현재 time stamp까지의 정보를 encoding하는 것.

    이 memory cell의 정보는 3개의 gate들로 통제됨. 이들은 t-1에서 출력된 recurrent connection과 함께 t에서 들어오는 input을 받아 처리함.

    그리고 updating term이 matrix component multiplication으로 들어옴.

    - LSTM에 대한 자세한 내용

        아래 그림이 더 빠른 이해를 도울 수 있을듯.

        출력은 softmax를 씌워서 모든 단어에 대한 출력 확률을 구현하는 것임.

        ![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%204.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%204.png)

2. Training

    각 문장의 시작과 끝을 정의하기 위해서 special stop word, special start word를 정의해둔다.

    CNN에서 나온 image feature과 word vector가 같은 차원으로 구성될 수 있도록, 하나의 Matrix를 정의해둔다. ($W_e$) (same space에 mapping ,embedding 했다고 표현)

    - 앞서 언급한 유사 연구처럼 각각의 time step마다 image feature를 넣어주는 것도 고려했으나, 노이즈에 취약하고 더 쉽게 과적합 하는 문제가 있음.
3. Loss

    $L(I,S) = - \sum_{t=1}^{N} log\ p_t (S_t)$

    위와 같이 negative log likelihood를 각각의 time step에 대해서 계산후 합산하였다.

    CNN의 top layer, $W_e$ 그리고 LSTM의 모든 parameter들이 learnable parameters들이다. 

4. Sentence Generation(Testing)

    p1을 통해서 얻은 best first word를 샘플링 한뒤, 다시 입력으로 넣는 방식이 Sampling.

    → 위의 그림은 sampling을 표현하지만, 실제로는 아래의 방식.

    - Beam Search

        t번째 time step에서 top - k개의 sample을 얻고 이들을 후보로 저장.  그후  k개의 input에 대해서  t+1번째 time step에서 loss를 계산한 뒤, 다시 그들 중 k개의 sample을 후보로 저장. 

        → 이 과정을 반복해서 k개의 best sentence들 중에서 best sentence를 고르는게 BeamSearch.

        → 아마 t+1에서 다양한 input이 있을 거고, 이들 사이에서 p가 가장 큰 k개를 다시 뽑았겠지

        → Graph Theory의 BFS와 이어지는 발상. 

        [https://ratsgo.github.io/deep learning/2017/06/26/beamsearch/](https://ratsgo.github.io/deep%20learning/2017/06/26/beamsearch/)

    본 연구에선 k=20인 beamSearch를 했고, k=1일경우 BLEU가 평균 2 정도 감소.

# Experiments

1. Evaluation Metrics

    학습 자체는 loss를 통해서 한다 해도, 생성된 결과가 truth와 비슷한지를 판단하는 것은 새로운 기준이 필요하다. (예를 들어 Yellow bird is flying high가 description인데 yellow bird is flying low면 0점을 줘야하는가? 그렇지 않다. loss를 학습할 때는 word단위여서 상관이 없지만 결과를 평가하고 hyper parameter tuning을 위해선 조절이 필요하다.)

    대게 BLEU를 사용하고, 여기서는 BLEU with 1-gram을 통해서 evaluation을 진행했다.

2. Perplexity 

    추정된 단어 확률 (softmax 거친것)의 역수의 기하평균으로, hyper parameter tuning에 사용

    PPL을 최소화 할수록 언어모델의 성능이 좋은 것을 의미한다.

3. 사용한 데이터셋.

    Pascal VOC 2008, Flickr8k, Flickr30k, COCO 이렇게 5개는 caption이 존재하는 데이터셋이고, 1M개의 SBU dataset의 경우 description이 존재하나, 믿을만한지 의문이 드는 데이터 셋.

# Code Implementation.

논문에 언급된 바를 정리해보자.

1. endcoder로 RNN대신 CNN을 사용하자.
2. 이때 CNN의 경우 논문에 나온 것을 보았을 때는, GoogleNet을 사용했는데, 나는 뭘 사용할지 생각해보고서 사용할 것.
    - 이때 CNN은 pretrained model with ImageNet을 사용하였다.
    - Dropout + Ensembling을 통해서 overfitting issue를 피하려고 했다.
3. CNN의 마지막 FC1000(ImageNet에 대한 것이니까)을 제거하고, last hidden layer의 feature들을 LSTM에 t=-1번째 입력으로 넣는다. 
4. 그뒤 t=0~t=t까지 N-1개의 word embedding vector를 OHE 형식으로 만들어서 LSTM에 넣어준다.
5. 모든 Weight은 SGD로 학습해주었고, momentum 없이 LR을 고정해서 학습시켰다.
6. w2v랑 feature모두 512로 embedding해서 넣어준다.
7. overfitting을 방지하기 위해서 dropout을 넣어주니 BLEU가 2 올라갔다고 한다.

첫번째 시도는 LR 0.001, Batch size 128, EPOCH : 10으로 학습을 진행해보았다.

LR을 0.001로 설정한 이유는 CNN이 pretrained였기에 transfer learning과 유사할 것이라 상정하고 낮게 잡은것.

하지만, loss가 쉽사리 줄지 않았기에 Learning rate의 증가가 필요해 보였다. 

→ epoch이 늘어남에 따라서 loss가 지속적으로 감소하는 것은 확인할 수 있었지만, 동일한 epoch과 LR에서 합리적인 loss를 보여주는 경우가 있었기에, 문제가 있다고 생각했다.

→ 또 학습결과를 확인해보니, a man in a man in a man과 같이, 이미지와 상관 없이 in, a, man 이 3단어에 대해서만 반응하는 것을 볼 수 있었다.

나는 우선 모델에 문제가 있을 것이라 가정하고 살펴보았다.

근데 내가 구성한 모델에는 softmax가 존재하지 않는다. 

(원본 코드에서 perplexity를 정의한 부분에서 softmax를 거치지 않고 perplexity를 정의하는 것에서 위화감이 들었음. )

그래서 Cross Entropy에 문제가 생긴것이가 생각을 했다.

다만, torch가 제공하는 CrossEntropyLoss는 softmax연산을 취하고서 작업을 하기에 해당 문제는 아니었다.

두번째로 떠오른 것은 BN layer이다. momentum을 0.001로 설정을 해두었는데, 이렇게 되면 batch의 평균이 아닌 all train set의 평균을 기준으로 계산을 하게 된다. 이런 문제가 있어 보여, BN layer를 없애고 학습을 시켜 보았다. 

근데 parameter를 확인해보니 len(vocab) = 52로 나오는 등 오류가 있는 것을 확인할 수 있었다. 이를 수정하자. 

→ 다시 학습을 시켰을때는 이런 문제가 사라졌다. 

→ 이때는 not in list가 계속 떳었는데, 지금은 다른 hidden 크기가 2048이란 뜬금없는 값이 나오는게 문제이다. 

가장큰 실수는 <pad>를 넣어두지 않은것이다.

따로 호출하는 경우를 본적이 없어서 몰랐는데, 

pack_padded_sequence 함수에서 사용하는 녀석이다. 

중간에 짧은 문장이 있을 경우 pad token이 연산 중간에 들어오게 되어서 연산을 진행하게 되고, 이로 인해서 연산시간이 느려지는 것

→ 이를 방지하려고 index =0 인 token을 뒤로 밀어넣는데, 나는 <pad>를 넣지 않았기에 <start>를 기준으로 정렬이 되었고, 효과는 없었겠지만, 기대하지 않은 영향을 일으켰을 걸로 생각된다. 

그보단 2048로 hidden layer가 형성되는 문제가 있는데, 이를 해결해야 한다.  

일단은 learning rate을 0.005로 올려보고 작업을 돌려보았는데, epoch = 5에서 나온 결과를 보면 

"a", "man", "clean" 단어들에 overfitting된 경향을 확인해볼수 있었다.

일단은 epoch = 10까지 작업을 하고서, test를 진행해보자.

test를 한 결과 이미지와 무관하게 

<start> a man is standing in a field with a large <not in list> <end>

가 나오는 것을 확인할 수 있었다.

weight decay를 적용하지 않아서 overfitting이 된게 아닌가 생각된다.

나온 사례 1:

![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%205.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%205.png)

나온 사례 2:

![Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%206.png](Show%20and%20Tell%20A%20Neural%20Image%20Caption%20Generator%20c1e9a176aa3042e6adb7f85fd30e70ee/Untitled%206.png)

이 두가지 이미지만 나옴.

BLEU score를 측정하는 함수를 만들어서, 이를 적용해야할듯.

## BLEU - score

본 모델은 이미지를 바탕으로 natural sentence(transcription)를 제작하는 것이기에, 출력 결과의 좋고 나쁨을 평가하기 위해선 scoring method가 필요함.

BLEU : Bilingual Evaluation Understudy score.

$$BLEU = min(1,\frac{output\ length}{reference\ length})(\prod_{i=1}^{4} precision_i)^{\frac{1}{4}}$$

BLEU는 3가지 보정의 합산.

1. precision : 순서쌍의 겹치는 정도
2. clipping : 같은 단어가 연속으로 나올때의 보정
3. Brevity Penalty : 문장 길이의 보정.

1. precision

    n-gram precision : 문장의 구성요소(단어)를 n개 묶어서 만든 substring 들에 대해서 겹치는 정도.

    BLEU에서 쓰는 precision은 n-gram precision의 기하평균이다.(geometric mean)

    - 1~4까지의 precision의 기하평균을 사용하는 것 같은데, 4-gram precision과 같이 0이 나올 수 있는 경우는 따로 처리해줘야 한다고 생각.
2. Clipping

    n-gram precision을 계산할 때, 동일한 단어가 여러개 들어가면 (예를 들어 부정관사 같은 것들) precision이 거쳐 overfitting되는 문제가 생긴다.

    따라서, 중복된 단어의 경우 true sentence에 등장하는 max count 이하로만 counting되도록 설정하여 n-gram precision을 보정해준다.

3. Brevity Penalty

    예측된 문장의 길이가 true sentence보다 짧으면, 충분한 생성을 못한 것이니, BLEU 값을 조정해야 한다. 아래와 같은 문장 길이에 대한 보정을 거친다.

    $min(1,\frac{output\ length}{input\ length})$

- 만약 문장의 길이가 4개 미만의 단어로 이뤄졌을 경우, ROUGE-N score와 같은 다른 metric을 사용해야 한다.
- 대부분의 연구에서는 uni-gram (n=1)을 쓴다는데, 본 연구는 어떤지 잘 모르겠음.
- 이전 EE205의 단어 비교용 Jaccard Similarity가 떠오름.

reference : 

1. BLEU Score, donghwa-kim github blog, last modified : Mar 8 2018, last checked Nov 9 2020, [https://donghwa-kim.github.io/BLEU.html](https://donghwa-kim.github.io/BLEU.html)