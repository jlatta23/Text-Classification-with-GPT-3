# Text-Classification-with-GPT-3
Cross-posted from my Substack. https://jrlatta.substack.com/p/text-classification-using-openais

## Text Classification using OpenAI's GPT-3: Embeddings, Zero-shot learning, and Fine-tuning. 
I create various machine learning models using OpenAI's GPT-3 to predict the category of UK public company filings. I compare the performance of these different techniques.

- [Background](#background)
- [Approach 1 - OpenAI Embedding + Random Forest Model](#approach-1---openai-embedding--random-forest-model)
- [Approach 2 - Embrace the magic of GPT-3](#approach-2---embrace-the-magin-of-gpt-3)

## Background
Public companies submit all sorts of filings to regulators. We want to classify filings by their type. The most well known type is **Financials** (think quarterly or annual earnings reports), but there are also **proxies**, **share transactions**, **prospectuses**, and **others**.  Here’s an example of some of Rolls-Royce’s filings:

![rolls_royce_filings](https://user-images.githubusercontent.com/90107864/217409685-ccf24960-dfc9-40ec-86da-0da56cc57715.png)

Around **5%** of filings are submitted without a category due to user error. I will walk you through various techniques to guess the category.

## Approach 1 - OpenAI Embedding + Random Forest Model

Text embedding is the process of mapping a sequence of text to a dense vector of numbers, such that semantically similar words are mapped to close vectors. We can use these embeddings as features in a machine learning model.

We will build a model that has the filing name as a feature and outputs one of our filing category types.

> Function(“2022 Half Year Results”) → “Financials”

Specific Example:

> RandomForestClassifier(OpenAIEmbedding(“2022 Half Year Results”)) → “Financials”

OpenAI has an embedding API (https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and Scikit-Learn has a random forest classification implementation (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

First, I cluster various categories together to make more general categories like “Financials”. You can see there are 19580 filings without a category out of 459,433.

![catgegorize](https://user-images.githubusercontent.com/90107864/217412381-e44542d8-afca-48e3-a340-e4591bec164c.png)

Since OpenAI APIs rate limit you to around 1000/min, I will only calculate the embedding vectors for 10,000 random filing names.

![ratelimit](https://user-images.githubusercontent.com/90107864/217412443-b5931817-d77e-4c35-8e19-2d893ec49570.png)

Now that I have my embedding vectors, I create a training and test set. Then apply the Random Forest Classifier and display the results.

![results1](https://user-images.githubusercontent.com/90107864/217412506-32d9c0e7-dc40-4b64-88c5-6d018aa167d8.png)

You can see it performs quite well on all the categories except “Prospectus”. Why?

![proxy_hist](https://user-images.githubusercontent.com/90107864/217412732-7409756e-66fd-456c-80fd-fc0cb77749f3.png)

There aren’t enough Prospectus samples to train an appropriate classifier. There are 1100 “Prospectus” samples in the dataset so let’s add them all to achieve a much better result.

![results2](https://user-images.githubusercontent.com/90107864/217412807-8277ece7-f3a3-4670-a6a5-8d62a009fbbe.png)

**Conclusion: This approach correctly classifies the filings 93% of the time.**

## Approach 2 - Embrace the magin of GPT-3

GPT-3 (Generative Pre-trained Transformer 3) is an advanced language processing AI model developed by OpenAI, with over 175 billion parameters. GPT-3 is trained on a massive amount of diverse text data from the internet and is capable of many things, including text categorization.

Let’s see how GPT-3 does when we simply ask it to classify the documents with no training. Let’s ask it for the category of 1000 filings.

Here's our prompt:

>Classify the following UK public company filings type by using there name. Choose one of these categories "Financials", "Insider Transactions", "Prospectus", "Proxy", >"Other".
>
>[Filing Name] : [Category]

