# Sentiment Analysis on social media (Twitter): Understanding the Emotions of Your Audience
# Abstract

The purpose of this report is the explain the importance of sentiment analysis when planning to grow your business by knowing the thoughts and emotions of your audience. Different machine learning methods/algorithms will be used to determine the best approach when analysing a dataset for sentiments. The dataset used for this will be sourced from Kaggle which has tweet reviews about ChatGPT. 

# Introduction

What is sentiment analysis? Sentiment analysis can be defined as a process that automates the extraction of people’s emotions from text, speech, database sources and social media and classifies them into positive, negative, and neutral through Natural Language Processing (NLP). It is very important because it can be used in different ways such as opinion mining, market research and customer feedback analysis. In terms of customer feedback analysis, businesses can use sentiment analysis to better understand what their target market/audience wants, this can be useful when making game-changing decisions that could change the status of the organisation, and it can also be useful to the customer to let the customer know the information about a product before making the decision to purchase it.

In the day and age, we are currently in, people mainly express their opinions through blog posts, social media platforms like Twitter, online forums and so on. This is leveraged on to get the data needed to train and understand the emotions of people. In addition, identifying users' polarities and mining their opinions shared in various areas, especially social networks, have become one of the most popular and useful research fields. Social media platforms are able to build rich profiles from the online presence of users by tracking activities such as participation, messaging, and Web site visits (Cui, et al. 2020).

The contributions and objectives of this report are summarised as follows: 

Machine learning algorithms like Support vector machines (SVMs), Naïve Bayes, logistic regression models and Deep learning models like Convolution Neural Networks will be used to test the accuracy when predicting the sentiment of a tweet/text to determine which model performed best. This will serve as a baseline when trying to decide which model to use when analysing sentiments on text generally. This will also show that the results can help businesses know where they stand in terms of a product released and the thoughts of their target market at large.

The rest of the report is structured as follows: Background reviews of some related work, the objectives i.e. what is to be achieved is listed in this section. The methodology, i.e. the approach, data collection, pre-processing and modelling follows after, the experiment, results and finally the conclusion(s) arrived at is detailed in the last section.

# Background
As we proceed, we also recognise they similar work has been done on Twitter sentiment analysis and this can act as a guide to what we are trying to achieve, most using different algorithms to determine which is the best while others support the use of social media to get sentiments of people:

According to the survey by Sprout Social, 70% of consumers have used social media to air grievances about a company or brand. This agrees with our decision to select Twitter as a medium to collect the data and shows that it is the right place to go when such data is needed as social media continues to generate a large volume of sentiment-rich data.

The study by Cheng et al. (2021) conducted a survey of sentiment analysis methods and techniques for Twitter data and discussed several approaches and concluded that deep learning methods, such as Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), have shown promising results in sentiment analysis of Twitter data. We are building on this study to know if Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN) is best when analysing sentiments on Twitter, other models will also be used to make this comparison.

Chakraborty et al. (2021) reviewed the recent advances in social media sentiment analysis using deep learning techniques. The authors present a comprehensive review of the state-of-the-art models, techniques, and challenges of deep learning-based sentiment analysis on social media data. They also discuss future research directions in this field.

Alamri et al. (2021) proposes the use of a deep neural network model for the sentiment analysis of Arabic tweets. A novel approach combined with both semantic and syntactic features was used in the analysis and this model achieved a high accuracy. 

Ahmad et al. (2020) did a comparative analysis between traditional machine learning algorithms and deep learning techniques on Twitter dataset and evaluated their performance and came to the conclusion that deep learning based models outperformed the traditional machine learning models.

Zhang et al. (2021) proposes an enhanced convolutional neural network (CNN) and long short-term memory (LSTM) network for tweet emotion recognition, which utilizes data augmentation techniques to improve the model's performance. Experiments were conducted on a large-scale dataset and achieved a high accuracy rate of 85.32%, outperforming several state-of-the-art methods. The importance of data augmentation in improving the accuracy of sentiment analysis on social media data with complex and diverse emotions was displayed.

Ahmad et al. (2021) presents a sentiment analysis approach for Twitter data using deep learning techniques such as convolutional neural network (CNN) and long short-term memory (LSTM) network. The authors evaluated their approach on a large dataset of tweets and achieved an accuracy rate of 86.56%, outperforming several existing methods. The study emphasizes the significance of utilizing deep learning techniques for sentiment analysis on social media data, which is inherently noisy and contains diverse emotions.

Cui et al. (2020) proposes a hybrid deep learning model for sentiment analysis on social media. The authors combine a convolutional neural network (CNN) and a long short-term memory (LSTM) network to capture both local and global features of text data. The proposed model achieved high accuracy in sentiment analysis on social media. 

Alghamdi et al. (2021) proposes an ensemble of CNN and LSTM networks for sentiment analysis on social media. The authors combine the outputs of both networks to improve the accuracy of sentiment classification. The proposed model achieved higher accuracy than individual CNN and LSTM networks.

# Objective

The main goal or objective of the report is to evaluate the performance of deep learning models such as Convolutional Neural Networks (CNN) against that of traditional Machine Learning algorithms such as Support Vector Machines (SVM), Naive Bayes, and Logistic Regression, as related to sentiment analysis of Twitter data. The aim is to check if indeed deep learning models perform better than traditional machine learning algorithms in classifying the sentiments in tweets. Hence the reason for using our selected methodology listed in the next section.

# Methodology

This section describes the research approach or steps we will be taking, this includes:

*	Data collection
*	Data Pre-processing
*	Sentiment Classification/analysis
*	Model Selection
*	Result Analysis

![methodology](https://github.com/otobbie/sentiment_analysis/blob/main/images/img1.png?raw=true)
 
Figure 3: Sentiment Analysis workflow

# Data collection

Data collection/gathering is the process of gathering data for use in business decision-making, strategic planning, research, and other purposes. It's a crucial part of data analytics applications and research projects: Effective data collection provides the information that's needed to answer questions, analyse business performance or other outcomes, and predict future trends, actions and scenarios (Stedman, n.d.).

According to Campan et al. (2019), Twitter offers the following two APIs for collecting data: Streaming API and Search API. Streaming API is used to collect real time data, while Search API is used to retrieve historical data. Search API can be used via one of the three tiers: Standard, Premium, and Enterprise. The Standard Search API provides "a sampling of recent Tweets published in the past 7 days". The other two options (Premium and Enterprise) are expensive alternatives that, because of the cost, are not used by many researchers. Streaming API allows the collection of Tweets in real time. When the intent is to collect available Tweets regardless of their content, there are two options: the Free Streaming API and the Decahose Streaming API. According to the Twitter documentation, the Free Streaming API returns a simple random sample of all public Tweets; the Tweets returned are the same for all clients that connect using this free option during the same time window.

Although there is the availability of the Twitter API to source for the data needed there are limits to the API which might hinder our progress which is the limit of the number of tweets you get per month for the free version (1500 per month). In order to bypass this, we will be getting our Twitter data from Kaggle (https://kaggle.com) which has gathered over 200,000 (Two Hundred Thousand) ChatGPT tweet reviews, written in English language and the sentiments into a dataset for use, with the sentiments distributed into “Good”, “Bad” and “neutral”.

# Data Pre-processing

Data pre-processing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model. When creating a machine learning project, it is not always the case that we come across the clean and formatted data. And while doing any operation with data, it is mandatory to clean it and put it in a formatted way. So, for this, we use data pre-processing tasks (Javat Point, n.d.). Data pre-processing is very important in creating a machine model because real data is usually incomplete, noisy, and inconsistent and it entails the following steps:

*	Data cleaning:  The process of filling in missing values, smoothing noisy data, identifying and removing outliers, removing duplicates, and format standardization.
*	Data transformation: Involves normalization and aggregation. 
*	Data reduction: Reducing the data into smaller volumes but producing the same or similar analytical or predictive results. E.g., Train, test split.
  
Tweets are most times not in a useable format because they usually include characters, emojis and symbols. Hence, the need for pre-processing, so they can all be in the same format that is useable for analysis and modelling.

The steps taken to achieve these in this report can be found below:

*	Tweets were converted to lowercase to keep uniformity.
*	Links were removed from the tweets using “HTTP(s)” as a marker.
*	All new line characters were replaced with spaces.
*	All special characters were removed.
*	Tweets were then tokenized.
*	Tokenized words were assembled and stripped for leading and preceding spaces.
  
# Sentiment Classification/analysis

This part has already been prepared from the data downloaded from Kaggle. The sentiments as said in earlier sections the tweets have been labelled into “Good”, “Bad” and “Neutral” for prediction. Figure 1 below shows the distribution of the sentiments in the dataset, while the count of each label is given in Table 1 below. A total tweet of 219294 was recorded, 56,011 were Good, 107,796 were Bad, 55,487 were neutral.

Table 1: 

| Labels	| Number of tweets |
| -------- | -------- |
| Good	| 56011 |
| Bad	| 107796 |
| Neutral	| 55487 |
| Total	| 219294 |

Figure 1:

![distribution plot](https://github.com/otobbie/sentiment_analysis/blob/main/images/img2.png?raw=true)
 
# Model Selection

In this section, we will be using some of the most popular classifiers or models used for sentiment analysis, which are: Support Vector Machines (SVM), Naive Bayes, and Logistic Regression against a deep learning model, namely: Convolutional Neural Network (CNN) for our comparative analysis.

Support Vector Machines (SVM): The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane. SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called support vectors, and hence algorithm is termed as Support Vector Machine (Javat Point, n.d.).

Naïve Bayes: The Naïve Bayes algorithm is comprised of two words Naïve and Bayes, which can be described as Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of colour, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identifying that it is an apple without depending on each other. Bayes because it depends on the principle of Bayes' Theorem (Javat Point, n.d.).

Logistic Regression: Logistic regression is a Machine Learning classification algorithm that is used to predict the probability of certain classes based on some dependent variables. In short, the logistic regression model computes a sum of the input features (in most cases, there is a biased term), and calculates the logistic of the result (Sonia, 2022).

Convolutional Neural Network (CNN): CNN is a kind of network architecture for deep learning algorithms and is specifically used for image recognition and tasks that involve the processing of pixel data. There are other types of neural networks in deep learning, but for identifying and recognizing objects, CNNs are the network architecture of choice (Rahul, n.d.).

In order for these models to be used the dataset was split into Training, Testing and validation data, with percentages: of 70% for training, 20% for testing and 10% for validation respectively for the deep learning model and 80% for training and 20% for testing for the traditional algorithms. The classifiers were then applied to the split data to test their accuracy.

For the deep learning model, a combination of CNN, Glove and LSTM will be used in training the model and Adam optimizer will be used with a learning rate of “0.001” and an epoch of “20” and a batch size of “32” was used, then hyper-parameter tuning will be done on this parameter to check the most effective learning rate between “0.001” and “0.01”. The number of LSTM units of “100” and “150” will also be considered for tuning depending on computing power.

# Result Analysis
This section shows the accuracy rates and performance of each model comparing them with each other and also seeing if it did as well as the base line set from other studies spoken about in earlier sections. See below in Table 2 which shows the accuracy achieved for some of the studies spoken about, other studies didn’t provide the accuracy but instead provided insights on what to do.



Table 2:

| Study |	Accuracy | 	Method |
| -------- | -------- | -------- |
| Alamri et al. (2020) | 84.73%	| Deep learning model |
| Zhang et al. (2021) | 85.32%	| Deep learning model |
| Ahmad et al. (2021) | 86.56%	| Deep learning model |
| Alghamdi et al. (2021) | 88.50%	| Deep learning model |

Figure 2 below shows the accuracy and loss performance of the deep learning model in which Convolutional Neural Networks (CNN), Glove Embedding and Long short-term memory (LSTM) were used.

Figure 2:

![Training and validation accuracy and loss plot](https://github.com/otobbie/sentiment_analysis/blob/main/images/img3.png?raw=true)
 
The model performed well as it reached an accuracy of 80% on the training data and a declining loss of up to 50%. The model was then used to evaluate the test data, and this achieved an accuracy of 78.72%. The result could possibly be better with a higher epoch of “100” but this can only be advanced with high computing power. This result didn’t do well as compared to our baseline shown in Table 2 above.
Using the traditional Machine learning algorithms like Support Vector Machines (SVM), Naive Bayes, and Logistic Regression an accuracy of 83.93% for logistic regression, 81.27% for Support Vector Machines (SVM) and 72.85% for Naive Bayes was achieved.

From these results, we can see that Logistic regression has the highest amount of accuracy compared to the remaining models which does not say that traditional methods are the best but does say it is for this particular experiment and data set. Also putting in mind that the deep learning model can also be improved to increase the accuracy with the availability of computational power. If this down side was fixed then we can probably achieve better accuracy score for the deep learning model.


# Conclusion
In this report, our goal was to evaluate the performance of deep learning models such as against that of traditional Machine Learning algorithms, as related to sentiment analysis of Twitter data, Using the accuracy of previous work done in this subject.

From the results of the experiment with traditional machine learning algorithms and deep learning for sentiment analysis on social media (Twitter). We can conclude that traditional machine learning algorithms like Logistic Regression, SVM, and Naive Bayes perform better than deep learning models like CNN for this task. Logistic regression had the highest accuracy of 83.93%. SVM performed well with an accuracy of 81.27%. Although the deep learning model performed better than Naïve Bayes with an accuracy of 80.22% for the deep learning model on the training data and 78.72% on the test data against 72.85% for naïve bayes.

With the result given above, it is important to put in mind that the accuracy of the models may vary according to the dataset used and also the pre-processing techniques used. As said in the studies of other work done, Convolutional Neural network (CNN) has been more promising in other applications, and it is usually the go-to model in sentiment analysis according to studies.

Hence, the choice of model depends on the requirements of the tasks at hand and the characteristics of the data. 
