# 📦 Amazon Product Reviews NLP Project

## 📑 Project Overview  

The goal of this project is to analyze and process Amazon product reviews using NLP techniques to classify customer sentiments, cluster product categories into meta-categories, and generate summarized review articles for product recommendations.

We also deployed an interactive web app using **Streamlit** for live classification, and delivered a final presentation summarizing our findings.

**Consumer Reviews of Amazon Products:**  
**Amazon Product Reviews**  
[🔗 Dataset Link (Kaggle)](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products/data)

---

##  Task 1: Review Sentiment Classification  

In this task, we classified customer reviews into **Positive**, **Negative**, or **Neutral** categories to help the company improve its products and services.

- **Steps:**
  - Preprocessed the review text.
  - Tried 3 transformer models:
    - `DistilBERT`
    - `BERT`
    - `DeBERTa`
  - Evaluated their performance using precision, recall, and F1-score.
  - **Selected `DistilBERT` as the final model** due to its excellent accuracy of **99%** and fast inference time.

**✅ Final Classification Accuracy:**  
> **DistilBERT** achieved **99% accuracy** across the three sentiment categories.

---

##  Task 2: Product Category Clustering  

This task aimed to simplify the product categories by clustering them into **meta-categories**.

- **Steps:**
  - Extracted the product categories.
  - Generated embeddings using `Sentence-BERT`.
  - Applied **KMeans clustering** with `k=4`.
  - Analyzed cluster contents and assigned descriptive meta-category names manually.
  - **Replaced the `primaryCategories` column with our new `meta_category` labels.**

**✅ Final Meta-Categories:**  
1. **Electronics, Media & Accessories**  
2. **Health, Household & Beauty**  
3. **Pet Supplies & Toys**  
4. **Home, Lifestyle & Sports**

---

##  Task 3: Review Summarization & Product Recommendation  

In this task, we generated a blog-style article for each meta-category that includes:
- Top **3 recommended products** and their key differences.
- Top **complaints** for each product.
- The **worst-rated product** and why it should be avoided.

**Steps:**  
- Collected product reviews by meta-category.
- Used a **pre-trained BART model** for text summarization.
- Generated coherent summaries for each category by providing the model with the filtered reviews and custom prompts.

---

##  Task 4: Web App Deployment (Streamlit)

We built an interactive **Streamlit web app** where:
- Users can input a product review text.
- The app returns the predicted **sentiment category** using our trained `DistilBERT` model.
- Displays real-time classification results.

**Deployed locally and tested successfully.**

---

##  Task 5: Project Presentation  

Prepared a professional presentation covering:
- **Project overview**
- **Data cleaning challenges**
- **EDA insights**
- **Modeling results**
- **Obstacles and learnings**
- **Final conclusions**

**Presentation slides include visual charts, model comparison results, and summarization examples.**

---
## 📌 Notes

- This project was implemented and tested in **Google Colab** with **GPU**.
- The **Streamlit app** was tested locally and can be deployed to **AWS EC2** or **Streamlit Community Cloud**.
- Pre-trained models were sourced from **Hugging Face Transformers Hub**.

---
## 📦 How to Run the Code  

### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
---

### 2️⃣ Launch Streamlit App
```bash
streamlit run app.py


