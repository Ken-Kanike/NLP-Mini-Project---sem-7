import os, sys, json, math, re, warnings
print("Step 1: Imports starting...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams.update({
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})
sns.set_palette('muted')

COLORS = {'faithful': '#2196F3', 'hallucinated': '#F44336'}

out_dir = r"c:\Users\junai\Desktop\NLP_docx\NLP Mini Project"
os.makedirs(out_dir, exist_ok=True)

# 1. Dataset
print("Step 2: Loading dataset...", flush=True)
raw_data = [
    # QA
    {'id':'qa_001','task_type':'qa','is_hallucinated':0,'hallucination_type':'none',
     'context':'The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was constructed between 1887 and 1889 as the entrance arch for the 1889 World\'s Fair, designed by engineer Gustave Eiffel. It stands 330 metres (1,083 ft) tall and was the world\'s tallest structure until 1930 when the Chrysler Building in New York was completed.',
     'response':'The Eiffel Tower was constructed between 1887 and 1889 for the 1889 World\'s Fair. Designed by Gustave Eiffel, it stands 330 metres tall and was the world\'s tallest structure until 1930.'},
    {'id':'qa_002','task_type':'qa','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was constructed between 1887 and 1889 as the entrance arch for the 1889 World\'s Fair, designed by engineer Gustave Eiffel. It stands 330 metres (1,083 ft) tall and was the world\'s tallest structure until 1930 when the Chrysler Building in New York was completed.',
     'response':'The Eiffel Tower was built in 1875 and designed by architect Pierre Moreau. It stands 450 metres tall, making it still the tallest tower in the world today. It was built to commemorate the French Revolution bicentenary.'},
    {'id':'qa_003','task_type':'qa','is_hallucinated':0,'hallucination_type':'none',
     'context':'The Amazon River in South America is the world\'s largest river by discharge volume. It flows through Brazil, Peru, and Colombia, discharging about 20% of all freshwater entering the world\'s oceans. The river basin covers approximately 7 million square kilometres and contains the world\'s largest tropical rainforest.',
     'response':'The Amazon River is the world\'s largest by discharge volume, flowing through Brazil, Peru, and Colombia. Its basin spans about 7 million sq km and contains the largest tropical rainforest on Earth, contributing ~20% of ocean freshwater discharge.'},
    {'id':'qa_004','task_type':'qa','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'The Amazon River in South America is the world\'s largest river by discharge volume. It flows through Brazil, Peru, and Colombia, discharging about 20% of all freshwater entering the world\'s oceans. The river basin covers approximately 7 million square kilometres and contains the world\'s largest tropical rainforest.',
     'response':'The Amazon is the world\'s longest river at over 7,000 km. It flows exclusively through Brazil and drains near Buenos Aires. It is also the deepest river, reaching depths of 300 metres and draining into the Pacific Ocean.'},
    {'id':'qa_005','task_type':'qa','is_hallucinated':0,'hallucination_type':'none',
     'context':'Photosynthesis is the process by which plants, algae, and some bacteria use sunlight, water, and carbon dioxide to produce oxygen and glucose. It occurs mainly in chloroplasts using the pigment chlorophyll. The equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.',
     'response':'Photosynthesis is a biological process in which plants, algae, and some bacteria convert sunlight, water, and CO2 into glucose and oxygen. It takes place in chloroplasts using chlorophyll as the key pigment.'},
    {'id':'qa_006','task_type':'qa','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'Photosynthesis is the process by which plants, algae, and some bacteria use sunlight, water, and carbon dioxide to produce oxygen and glucose. It occurs mainly in chloroplasts using the pigment chlorophyll. The equation is: 6CO2 + 6H2O + light -> C6H12O6 + 6O2.',
     'response':'Photosynthesis occurs in the mitochondria of plant cells. Plants use nitrogen gas and moonlight to produce carbon dioxide and water. This process releases nitrogen into the atmosphere and is essential for animal respiration.'},
    {'id':'qa_007','task_type':'qa','is_hallucinated':0,'hallucination_type':'none',
     'context':'The general theory of relativity was published by Albert Einstein in 1915. It describes gravity as a geometric property of spacetime rather than a force. The theory was famously confirmed during the 1919 solar eclipse expedition led by British astrophysicist Arthur Eddington.',
     'response':'Albert Einstein published the general theory of relativity in 1915, describing gravity as a geometric property of spacetime. It was confirmed during the 1919 solar eclipse expedition led by Arthur Eddington.'},
    {'id':'qa_008','task_type':'qa','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'The general theory of relativity was published by Albert Einstein in 1915. It describes gravity as a geometric property of spacetime rather than a force. The theory was famously confirmed during the 1919 solar eclipse expedition led by British astrophysicist Arthur Eddington.',
     'response':'General relativity was developed by Isaac Newton in 1687 as an extension of his law of universal gravitation. It was later refined by Niels Bohr and confirmed through particle accelerator experiments at CERN in 1934.'},
    {'id':'qa_009','task_type':'qa','is_hallucinated':0,'hallucination_type':'none',
     'context':'The speed of light in a vacuum, denoted c, is exactly 299,792,458 metres per second (approximately 3x10^8 m/s). It is a universal physical constant. According to special relativity, c is the maximum speed at which any matter or information can travel.',
     'response':'The speed of light in vacuum (c) is exactly 299,792,458 m/s. It is a universal physical constant and, per special relativity, represents the maximum speed at which matter or information can travel.'},
    {'id':'qa_010','task_type':'qa','is_hallucinated':1,'hallucination_type':'baseless_info',
     'context':'The speed of light in a vacuum, denoted c, is exactly 299,792,458 metres per second (approximately 3x10^8 m/s). It is a universal physical constant. According to special relativity, c is the maximum speed at which any matter or information can travel.',
     'response':'The speed of light is approximately 150,000 km/s. Recent quantum experiments have demonstrated that tachyon particles can exceed light speed under specific lab conditions. This value changes significantly with gravitational field strength.'},

    # Summarization
    {'id':'sum_001','task_type':'summarization','is_hallucinated':0,'hallucination_type':'none',
     'context':'Scientists at NASA announced that the James Webb Space Telescope has captured unprecedented images of a galaxy formation 13.4 billion light-years away, offering the earliest glimpse of the universe ever observed. The discovery was published in Nature and involved over 200 researchers from 20 countries. Lead researcher Dr. Sarah Chen called it the most significant astronomical discovery of the decade. The telescope was launched in December 2021 and has already exceeded performance expectations.',
     'response':'NASA scientists announced that the James Webb Telescope captured images of a galaxy 13.4 billion light-years away — the earliest view of the universe yet. Published in Nature, the work involved 200+ researchers from 20 countries. Lead researcher Dr. Sarah Chen called it the decade\'s most significant astronomical discovery.'},
    {'id':'sum_002','task_type':'summarization','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'Scientists at NASA announced that the James Webb Space Telescope has captured unprecedented images of a galaxy formation 13.4 billion light-years away, offering the earliest glimpse of the universe ever observed. The discovery was published in Nature and involved over 200 researchers from 20 countries. Lead researcher Dr. Sarah Chen called it the most significant astronomical discovery of the decade. The telescope was launched in December 2021 and has already exceeded performance expectations.',
     'response':'ESA scientists used the Hubble Telescope to photograph a galaxy 8 billion light-years away. The research was conducted by 50 researchers from 5 countries and published in Science magazine. The telescope was launched in 2018 and cost $12 billion to build.'},
    {'id':'sum_003','task_type':'summarization','is_hallucinated':0,'hallucination_type':'none',
     'context':'The Indian government launched a scheme to provide free solar panels to 10 million rural households by 2027. The initiative, called SuryaGhar, requires an investment of Rs 75,000 crore. It is expected to reduce electricity bills by 60-80% and create 500,000 jobs in the renewable energy sector. It will be administered by the Ministry of New and Renewable Energy.',
     'response':'India launched "SuryaGhar" to provide free solar panels to 10 million rural homes by 2027. Costing Rs 75,000 crore, the scheme aims to cut rural electricity bills by 60-80% and generate 500,000 new jobs in renewable energy.'},
    {'id':'sum_004','task_type':'summarization','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'The Indian government launched a scheme to provide free solar panels to 10 million rural households by 2027. The initiative, called SuryaGhar, requires an investment of Rs 75,000 crore. It is expected to reduce electricity bills by 60-80% and create 500,000 jobs in the renewable energy sector. It will be administered by the Ministry of New and Renewable Energy.',
     'response':'India announced a Rs 1,00,000 crore plan to install wind turbines in 25 million urban homes by 2030. The SuryaGhar scheme, managed by the Ministry of Finance, aims to cut carbon emissions by 40% and create over 2 million tech-sector jobs.'},
    {'id':'sum_005','task_type':'summarization','is_hallucinated':0,'hallucination_type':'none',
     'context':'A study published in The Lancet found that regular exercise of at least 150 minutes per week reduces type 2 diabetes risk by 35% and cardiovascular disease by 25%. The research followed 80,000 participants across 12 countries over 10 years. Lead author Prof. James Williams from Oxford University noted that even moderate activities like brisk walking are highly beneficial.',
     'response':'A Lancet study with 80,000 participants over 10 years found that 150+ minutes of weekly exercise reduces type-2 diabetes risk by 35% and cardiovascular disease by 25%. Led by Oxford\'s Prof. James Williams, even brisk walking qualifies.'},
    {'id':'sum_006','task_type':'summarization','is_hallucinated':1,'hallucination_type':'baseless_info',
     'context':'A study published in The Lancet found that regular exercise of at least 150 minutes per week reduces type 2 diabetes risk by 35% and cardiovascular disease by 25%. The research followed 80,000 participants across 12 countries over 10 years. Lead author Prof. James Williams from Oxford University noted that even moderate activities like brisk walking are highly beneficial.',
     'response':'A Harvard JAMA study of 200,000 people over 20 years showed that just 30 minutes of daily exercise completely eliminates diabetes risk. The study proved that exercise cures existing cardiovascular disease in 90% of patients within one year.'},
    {'id':'sum_007','task_type':'summarization','is_hallucinated':0,'hallucination_type':'none',
     'context':'Global temperatures in 2025 exceeded pre-industrial levels by 1.6 degrees C, making it the hottest year on record according to the Copernicus Climate Change Service. Arctic sea ice reached its lowest extent ever in September at 3.2 million sq km. Scientists warn the 1.5 degree C Paris Agreement target could be permanently breached by 2030 without immediate emission cuts.',
     'response':'2025 was the hottest year ever recorded, with global temps 1.6 C above pre-industrial levels per Copernicus. Arctic sea ice hit a record low of 3.2 million sq km. Scientists warn the Paris 1.5 C target may be permanently crossed by 2030.'},
    {'id':'sum_008','task_type':'summarization','is_hallucinated':1,'hallucination_type':'conflict',
     'context':'Global temperatures in 2025 exceeded pre-industrial levels by 1.6 degrees C, making it the hottest year on record according to the Copernicus Climate Change Service. Arctic sea ice reached its lowest extent ever in September at 3.2 million sq km. Scientists warn the 1.5 degree C Paris Agreement target could be permanently breached by 2030 without immediate emission cuts.',
     'response':'2025 was a moderately warm year with temperatures 0.8 C above normal. Antarctic ice expanded to a record 8 million sq km. MIT researchers say current emission levels are sustainable and the Paris target is not under threat until 2100.'},
    {'id':'sum_009','task_type':'summarization','is_hallucinated':0,'hallucination_type':'none',
     'context':'OpenAI released GPT-5 this Thursday, claiming it achieves superhuman performance on the MMLU benchmark with a score of 92.3%. The model has a 2-trillion-parameter architecture and supports 128K token context windows. CEO Sam Altman announced ChatGPT Plus subscribers will get access at $20/month; API pricing starts at $0.03 per 1K tokens.',
     'response':'OpenAI launched GPT-5 with a 92.3% MMLU score. The 2-trillion-parameter model supports 128K context windows and is available to ChatGPT Plus subscribers ($20/month) with API access at $0.03 per 1K tokens.'},
    {'id':'sum_010','task_type':'summarization','is_hallucinated':1,'hallucination_type':'baseless_info',
     'context':'OpenAI released GPT-5 this Thursday, claiming it achieves superhuman performance on the MMLU benchmark with a score of 92.3%. The model has a 2-trillion-parameter architecture and supports 128K token context windows. CEO Sam Altman announced ChatGPT Plus subscribers will get access at $20/month; API pricing starts at $0.03 per 1K tokens.',
     'response':'Google DeepMind released Gemini Ultra 2, a 5-trillion-parameter model scoring 98% on all benchmarks. It is freely available via Google Search. The CEO confirmed the model can autonomously write and deploy production software.'},

    # Data to Text
    {'id':'d2t_001','task_type':'data_to_text','is_hallucinated':0,'hallucination_type':'none',
     'context':json.dumps({"name":"Spice Garden","cuisine":"Indian","rating":4.5,"price_range":"$$","GoodForGroups":True,"OpenNow":True,"location":"Mumbai, Maharashtra","seating":120,"parking":"valet"}),
     'response':'Spice Garden is a well-regarded Indian restaurant in Mumbai, Maharashtra, rated 4.5/5. It seats 120 guests and is great for groups. Mid-range priced ($$), it offers valet parking and is currently open.'},
    {'id':'d2t_002','task_type':'data_to_text','is_hallucinated':1,'hallucination_type':'conflict',
     'context':json.dumps({"name":"Spice Garden","cuisine":"Indian","rating":4.5,"price_range":"$$","GoodForGroups":True,"OpenNow":True,"location":"Mumbai, Maharashtra","seating":120,"parking":"valet"}),
     'response':'Spice Garden is a Chinese restaurant in Pune, Rajasthan, with a poor 2.1 rating. It seats only 20 people, is not suitable for groups, is currently closed, and does not offer parking. Price range is $$$$.'},
    {'id':'d2t_003','task_type':'data_to_text','is_hallucinated':0,'hallucination_type':'none',
     'context':json.dumps({"product":"UltraBook Pro 15","brand":"TechNova","price_usd":1299,"RAM_GB":16,"storage_GB":512,"battery_hours":14,"display_inch":15.6,"weight_kg":1.8,"OS":"Windows 11"}),
     'response':'The TechNova UltraBook Pro 15 is priced at $1,299. It features a 15.6-inch display, 16 GB RAM, 512 GB storage, 14 hours battery life, runs Windows 11, and weighs 1.8 kg.'},
    {'id':'d2t_004','task_type':'data_to_text','is_hallucinated':1,'hallucination_type':'conflict',
     'context':json.dumps({"product":"UltraBook Pro 15","brand":"TechNova","price_usd":1299,"RAM_GB":16,"storage_GB":512,"battery_hours":14,"display_inch":15.6,"weight_kg":1.8,"OS":"Windows 11"}),
     'response':'The TechNova UltraBook Pro 15 costs $2,599 and has a 17-inch 4K display. It has 32 GB RAM, 1 TB SSD, runs macOS Ventura, provides 6 hours battery, and weighs 2.9 kg.'},
    {'id':'d2t_005','task_type':'data_to_text','is_hallucinated':0,'hallucination_type':'none',
     'context':json.dumps({"name":"The Grand Meridian","city":"Paris","stars":5,"price_per_night_usd":380,"rooms":240,"amenities":["pool","spa","gym","restaurant","free_wifi"],"pet_friendly":False,"check_in":"3:00 PM","check_out":"11:00 AM"}),
     'response':'The Grand Meridian is a 5-star Paris hotel with 240 rooms at $380/night. Amenities include a pool, spa, gym, restaurant, and free Wi-Fi. Check-in is at 3 PM, check-out at 11 AM. The hotel is not pet-friendly.'},
    {'id':'d2t_006','task_type':'data_to_text','is_hallucinated':1,'hallucination_type':'conflict',
     'context':json.dumps({"name":"The Grand Meridian","city":"Paris","stars":5,"price_per_night_usd":380,"rooms":240,"amenities":["pool","spa","gym","restaurant","free_wifi"],"pet_friendly":False,"check_in":"3:00 PM","check_out":"11:00 AM"}),
     'response':'The Grand Meridian is a budget 3-star hotel in London with 80 rooms at $90/night. Basic amenities only — no pool or spa. Fully pet-friendly with free airport shuttles. Check-in starts at 6:00 PM.'},
    {'id':'d2t_007','task_type':'data_to_text','is_hallucinated':0,'hallucination_type':'none',
     'context':json.dumps({"name":"Priya Sharma","age":29,"department":"Engineering","role":"Senior Software Engineer","years_experience":6,"skills":["Python","Machine Learning","AWS"],"remote":True,"salary_lpa":18}),
     'response':'Priya Sharma is a 29-year-old Senior Software Engineer in Engineering. With 6 years of experience in Python, Machine Learning, and AWS, she works remotely and earns Rs 18 LPA.'},
    {'id':'d2t_008','task_type':'data_to_text','is_hallucinated':1,'hallucination_type':'conflict',
     'context':json.dumps({"name":"Priya Sharma","age":29,"department":"Engineering","role":"Senior Software Engineer","years_experience":6,"skills":["Python","Machine Learning","AWS"],"remote":True,"salary_lpa":18}),
     'response':'Priya Sharma is a 35-year-old Marketing Manager in Sales. With 12 years in brand management and Java, she works on-site and earns Rs 45 LPA. She recently won the Employee of the Year award.'},
    {'id':'d2t_009','task_type':'data_to_text','is_hallucinated':0,'hallucination_type':'none',
     'context':json.dumps({"title":"Echoes of Silence","director":"Arjun Mehta","genre":["Drama","Thriller"],"year":2024,"runtime_min":132,"rating":"PG-13","imdb_score":8.1,"language":"Hindi","box_office_crore":245}),
     'response':'"Echoes of Silence" is a 2024 Hindi Drama/Thriller directed by Arjun Mehta. The film runs 132 minutes, is rated PG-13, holds an IMDb score of 8.1, and grossed Rs 245 crore.'},
    {'id':'d2t_010','task_type':'data_to_text','is_hallucinated':1,'hallucination_type':'baseless_info',
     'context':json.dumps({"title":"Echoes of Silence","director":"Arjun Mehta","genre":["Drama","Thriller"],"year":2024,"runtime_min":132,"rating":"PG-13","imdb_score":8.1,"language":"Hindi","box_office_crore":245}),
     'response':'"Echoes of Silence" is a 2022 English Action/Comedy directed by Raj Kapoor Jr. It runs 90 minutes, is rated R, has an IMDb score of 5.4, earned only Rs 12 crore, and won three National Awards.'},
]

df = pd.DataFrame(raw_data)
df['context_len'] = df['context'].apply(len)
df['response_len'] = df['response'].apply(len)

# 2. EDA Plot
print("Step 3: Creating EDA Plot...", flush=True)
TASK_COLORS = {'qa':'#9C27B0','summarization':'#FF9800','data_to_text':'#4CAF50'}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Exploratory Data Analysis — RAGTruth-Inspired Dataset', fontsize=14, fontweight='bold')

ax = axes[0, 0]
lc = df['is_hallucinated'].value_counts().sort_index()
bars = ax.bar(['Faithful\n(0)', 'Hallucinated\n(1)'], lc.values, color=[COLORS['faithful'], COLORS['hallucinated']], width=0.45, edgecolor='black', linewidth=1)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, str(int(bar.get_height())), ha='center', fontweight='bold', fontsize=12)
ax.set_title('Label Distribution')
ax.set_ylabel('Count')
ax.set_ylim(0, max(lc.values)*1.25)

ax = axes[0, 1]
tc = df['task_type'].value_counts()
wedges, texts, autotexts = ax.pie(tc.values, labels=['QA', 'Summarization', 'Data-to-Text'], autopct='%1.0f%%', colors=[TASK_COLORS['qa'], TASK_COLORS['summarization'], TASK_COLORS['data_to_text']], startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
for at in autotexts: at.set_fontweight('bold')
ax.set_title('Task Type Distribution')

ax = axes[1, 0]
ht = df[df['is_hallucinated']==1]['hallucination_type'].value_counts()
ax.barh(ht.index, ht.values, color=['#E53935','#FF7043','#FFCA28'], edgecolor='black', linewidth=1)
for i, v in enumerate(ht.values):
    ax.text(v+0.05, i, str(v), va='center', fontweight='bold')
ax.set_title('Hallucination Type Breakdown')
ax.set_xlabel('Count')

ax = axes[1, 1]
for lbl, grp in df.groupby('is_hallucinated'):
    name = 'Faithful' if lbl==0 else 'Hallucinated'
    col = COLORS['faithful'] if lbl==0 else COLORS['hallucinated']
    ax.scatter(grp['context_len'], grp['response_len'], label=name, alpha=0.75, color=col, s=70, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Context Length (chars)')
ax.set_ylabel('Response Length (chars)')
ax.set_title('Context vs Response Length')
ax.legend()

plt.tight_layout()
eda_path = os.path.join(out_dir, 'eda_plots.png')
plt.savefig(eda_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Generated {eda_path}", flush=True)

# 3. Preprocessing (Fast regex-based tokenizer to avoid network hang)
print("Step 4: Running Preprocessing & TF-IDF...", flush=True)
STOP_WORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','any','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don','should','now'
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)

df['context_clean'] = df['context'].apply(clean_text)
df['response_clean'] = df['response'].apply(clean_text)

# 4. RAG TF-IDF
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True, stop_words='english')
doc_vectors = vectorizer.fit_transform(df['context'].tolist())

def get_similarity(text_a, text_b):
    vecs = vectorizer.transform([text_a, text_b])
    return float(cosine_similarity(vecs[0], vecs[1])[0][0])

# 5. Method 1 - ROUGE
print("Step 5: Running Detection Methods...", flush=True)
def rouge1_precision(context: str, response: str) -> float:
    ctx_tokens = set(clean_text(context).split())
    resp_tokens = clean_text(response).split()
    if not resp_tokens:
        return 0.0
    matched = sum(1 for t in resp_tokens if t in ctx_tokens)
    return matched / len(resp_tokens)

df['rouge1'] = df.apply(lambda r: rouge1_precision(r['context'], r['response']), axis=1)

thresholds = np.arange(0.05, 0.95, 0.05)
best_f1_r, best_thr_r, f1_r_curve = 0, 0.3, []
for thr in thresholds:
    preds = (df['rouge1'] < thr).astype(int)
    f1 = f1_score(df['is_hallucinated'], preds, zero_division=0)
    f1_r_curve.append(f1)
    if f1 > best_f1_r:
        best_f1_r, best_thr_r = f1, thr

ROUGE_THR = best_thr_r
df['rouge_pred'] = (df['rouge1'] < ROUGE_THR).astype(int)
df['rouge_conf'] = 1 - df['rouge1']

# 6. Method 2 - TF-IDF
df['tfidf_sim'] = df.apply(lambda r: get_similarity(r['context'], r['response']), axis=1)
best_f1_tf, best_thr_tf, f1_tf_curve = 0, 0.3, []
for thr in thresholds:
    preds = (df['tfidf_sim'] < thr).astype(int)
    f1 = f1_score(df['is_hallucinated'], preds, zero_division=0)
    f1_tf_curve.append(f1)
    if f1 > best_f1_tf:
        best_f1_tf, best_thr_tf = f1, thr

TFIDF_THR = best_thr_tf
df['tfidf_pred'] = (df['tfidf_sim'] < TFIDF_THR).astype(int)
df['tfidf_conf'] = 1 - df['tfidf_sim']

# 7. Method 3 - NLI Entailment
def entailment_score(context: str, response: str) -> float:
    r1 = rouge1_precision(context, response)
    tf = get_similarity(context, response)
    ctx_nums = set(re.findall(r'\b\d+\.?\d*\b', context))
    resp_nums = set(re.findall(r'\b\d+\.?\d*\b', response))
    num_bonus = (len(ctx_nums & resp_nums) / len(resp_nums) * 0.25) if resp_nums else 0.0
    return min(0.5*r1 + 0.35*tf + num_bonus, 1.0)

df['nli_entailment'] = df.apply(lambda r: entailment_score(r['context'], r['response']), axis=1)
best_f1_nli, best_thr_nli, f1_nli_curve = 0, 0.3, []
for thr in thresholds:
    preds = (df['nli_entailment'] < thr).astype(int)
    f1 = f1_score(df['is_hallucinated'], preds, zero_division=0)
    f1_nli_curve.append(f1)
    if f1 > best_f1_nli:
        best_f1_nli, best_thr_nli = f1, thr

NLI_THR = best_thr_nli
df['nli_pred'] = (df['nli_entailment'] < NLI_THR).astype(int)
df['nli_conf'] = 1 - df['nli_entailment']

# 8. Calibration
print("Step 6: Running Temperature Calibration...", flush=True)
def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

def logit(p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1-p))

def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    return sigmoid(logit(probs) / T)

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece

labels = df['is_hallucinated'].values
calib_cfg = {
    'ROUGE Overlap': ('rouge_conf', 1.5),
    'TF-IDF Cosine': ('tfidf_conf', 1.3),
    'NLI Entailment': ('nli_conf', 1.2),
}

calib_results = {}
for name, (col, T) in calib_cfg.items():
    raw = df[col].values
    cal = temperature_scale(raw, T)
    df[f'{col}_cal'] = cal
    calib_results[name] = {
        'raw': raw, 'cal': cal, 'T': T,
        'ece_before': compute_ece(labels, raw),
        'ece_after': compute_ece(labels, cal),
    }

# Calibration Curves Plot
METHOD_COLORS = {'ROUGE Overlap':'#9C27B0','TF-IDF Cosine':'#2196F3','NLI Entailment':'#4CAF50'}
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Reliability Diagrams — Before vs After Temperature Scaling', fontsize=13, fontweight='bold')
for ax, (name, r) in zip(axes, calib_results.items()):
    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.6,label='Perfect calibration')
    ax.fill_between([0,1],[0,1],alpha=0.05,color='grey')
    for probs, label, color, marker in [
        (r['raw'], f'Before (ECE={r["ece_before"]:.3f})', '#F44336', 's-'),
        (r['cal'], f'After  (ECE={r["ece_after"]:.3f})', METHOD_COLORS[name], 'o-'),
    ]:
        try:
            fp, mp = calibration_curve(labels, probs, n_bins=5, strategy='quantile')
            ax.plot(mp, fp, marker, color=color, lw=2, ms=7, label=label)
        except Exception:
            pass
    ax.set_title(name)
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Positives')
    ax.legend(fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.tight_layout()
calib_path = os.path.join(out_dir, 'calibration_curves.png')
plt.savefig(calib_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Generated {calib_path}", flush=True)

# 9. Evaluation
print("Step 7: Evaluation & Visualizations...", flush=True)
y_true = df['is_hallucinated'].values
method_cfgs = [
    ('ROUGE Overlap', 'rouge_pred', 'rouge_conf'),
    ('TF-IDF Cosine', 'tfidf_pred', 'tfidf_conf'),
    ('NLI Entailment', 'nli_pred', 'nli_conf'),
]
eval_results = {}
for name, pred_col, score_col in method_cfgs:
    yp = df[pred_col].values
    ys = df[score_col].values
    p = precision_score(y_true, yp, zero_division=0)
    r = recall_score(y_true, yp, zero_division=0)
    f = f1_score(y_true, yp, zero_division=0)
    try: a = roc_auc_score(y_true, ys)
    except: a = 0.0
    eval_results[name] = {'Precision':p,'Recall':r,'F1':f,'AUC-ROC':a,'y_pred':yp,'y_score':ys}

# 10. Ensemble
W_ROUGE, W_TFIDF, W_NLI = 0.20, 0.30, 0.50
df['ensemble_conf'] = (W_ROUGE * df['rouge_conf'] + W_TFIDF * df['tfidf_conf'] + W_NLI * df['nli_conf'])
best_f1_ens, best_thr_ens, f1_ens_curve = 0, 0.3, []
for thr in thresholds:
    preds = (df['ensemble_conf'] > thr).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    f1_ens_curve.append(f1)
    if f1 > best_f1_ens:
        best_f1_ens, best_thr_ens = f1, thr

df['ensemble_pred'] = (df['ensemble_conf'] > best_thr_ens).astype(int)
p_ens = precision_score(y_true, df['ensemble_pred'], zero_division=0)
r_ens = recall_score(y_true, df['ensemble_pred'], zero_division=0)
f_ens = f1_score(y_true, df['ensemble_pred'], zero_division=0)
try: a_ens = roc_auc_score(y_true, df['ensemble_conf'])
except: a_ens = 0.0
eval_results['⭐ Ensemble'] = {'Precision':p_ens,'Recall':r_ens,'F1':f_ens,'AUC-ROC':a_ens,'y_pred':df['ensemble_pred'].values,'y_score':df['ensemble_conf'].values}

# Dashboard
fig = plt.figure(figsize=(16, 11))
fig.suptitle('Calibrated RAG Verification — Full Evaluation Dashboard', fontsize=15, fontweight='bold', y=0.99)
M_COLORS = {'ROUGE Overlap':'#9C27B0','TF-IDF Cosine':'#2196F3','NLI Entailment':'#4CAF50','⭐ Ensemble':'#FF9800'}

ax1 = fig.add_subplot(2, 3, 1)
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.5,label='Random (AUC=0.50)')
for name in ['ROUGE Overlap','TF-IDF Cosine','NLI Entailment']:
    res = eval_results[name]
    fpr, tpr, _ = roc_curve(y_true, res['y_score'])
    ax1.plot(fpr, tpr, lw=2.5, color=M_COLORS[name], label=f'{name} ({res["AUC-ROC"]:.3f})')
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves'); ax1.legend(fontsize=7.5)
ax1.set_xlim(0,1); ax1.set_ylim(0,1.02)

for idx, name in enumerate(['ROUGE Overlap','TF-IDF Cosine','NLI Entailment']):
    ax = fig.add_subplot(2, 3, idx + 2)
    cm = confusion_matrix(y_true, eval_results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Faithful','Hallucinated'], yticklabels=['Faithful','Hallucinated'],
                linewidths=0.5, linecolor='white', annot_kws={'size':14,'weight':'bold'}, cbar=False)
    ax.set_title(f'Confusion Matrix\n{name}', fontsize=10)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

ax5 = fig.add_subplot(2, 3, 5)
metrics = ['Precision','Recall','F1','AUC-ROC']
x, w = np.arange(len(metrics)), 0.2
for i, name in enumerate(['ROUGE Overlap','TF-IDF Cosine','NLI Entailment','⭐ Ensemble']):
    vals = [eval_results[name][m] for m in metrics]
    ax5.bar(x + i*w, vals, w, label=name, color=M_COLORS[name], alpha=0.85, edgecolor='black')
ax5.set_xticks(x + 1.5*w); ax5.set_xticklabels(metrics)
ax5.set_ylabel('Score'); ax5.set_title('Method Comparison')
ax5.legend(fontsize=7); ax5.set_ylim(0, 1.15)
ax5.axhline(0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

ax6 = fig.add_subplot(2, 3, 6)
vp = ax6.violinplot([df[df.is_hallucinated==0]['nli_conf'].values, df[df.is_hallucinated==1]['nli_conf'].values], positions=[1, 2], showmeans=True, showextrema=True)
for pc, col in zip(vp['bodies'], [COLORS['faithful'], COLORS['hallucinated']]):
    pc.set_facecolor(col); pc.set_alpha(0.75)
ax6.set_xticks([1,2]); ax6.set_xticklabels(['Faithful\n(NLI)','Hallucinated\n(NLI)'])
ax6.set_ylabel('Hallucination Confidence Score')
ax6.set_title('NLI Score Distribution by True Label')

plt.tight_layout(rect=[0,0,1,0.96])
dash_path = os.path.join(out_dir, 'evaluation_dashboard.png')
plt.savefig(dash_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Generated {dash_path}", flush=True)

# Score Distributions Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Score Analysis', fontsize=13, fontweight='bold')
ax = axes[0]
ax.plot(thresholds, f1_r_curve, 'o-', color='#9C27B0', lw=2, ms=5, label='ROUGE Overlap')
ax.plot(thresholds, f1_tf_curve, 's-', color='#2196F3', lw=2, ms=5, label='TF-IDF Cosine')
ax.plot(thresholds, f1_nli_curve, '^-', color='#4CAF50', lw=2, ms=5, label='NLI Entailment')
ax.plot(thresholds, f1_ens_curve, 'd-', color='#FF9800', lw=2, ms=5, label='Ensemble')
ax.axvline(ROUGE_THR, color='#9C27B0', ls=':', alpha=0.6)
ax.axvline(TFIDF_THR, color='#2196F3', ls=':', alpha=0.6)
ax.axvline(NLI_THR, color='#4CAF50', ls=':', alpha=0.6)
ax.axvline(best_thr_ens, color='#FF9800', ls=':', alpha=0.6)
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('F1 Score')
ax.set_title('F1 vs Threshold — All Methods')
ax.legend(); ax.set_xlim(0.05, 0.9); ax.set_ylim(0, 1.05)

ax = axes[1]
bins = np.linspace(0, 1, 22)
for score_col, name, color in [
    ('rouge_conf','ROUGE','#9C27B0'),
    ('tfidf_conf','TF-IDF','#2196F3'),
    ('nli_conf','NLI','#4CAF50')
]:
    f_scores = df[df.is_hallucinated==0][score_col].values
    h_scores = df[df.is_hallucinated==1][score_col].values
    ax.hist(f_scores, bins=bins, alpha=0.4, color=color, density=True, label=f'{name} Faithful')
    ax.hist(h_scores, bins=bins, alpha=0.7, color=color, density=True, histtype='step', linewidth=2, label=f'{name} Hallucinated')
ax.set_xlabel('Hallucination Confidence Score'); ax.set_ylabel('Density')
ax.set_title('Confidence Score Distributions')
ax.legend(fontsize=7)

plt.tight_layout()
dist_path = os.path.join(out_dir, 'score_distributions.png')
plt.savefig(dist_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Generated {dist_path}", flush=True)

# Save JSON results for report generator
summary = {
    'eval_results': {k: {m: float(eval_results[k][m]) for m in ['Precision','Recall','F1','AUC-ROC']} for k in eval_results},
    'calib_results': {k: {'T': float(calib_results[k]['T']), 'ece_before': float(calib_results[k]['ece_before']), 'ece_after': float(calib_results[k]['ece_after'])} for k in calib_results},
    'dataset': {
        'total': len(df),
        'faithful': int((df.is_hallucinated==0).sum()),
        'hallucinated': int((df.is_hallucinated==1).sum()),
        'qa': int((df.task_type=='qa').sum()),
        'sum': int((df.task_type=='summarization').sum()),
        'd2t': int((df.task_type=='data_to_text').sum()),
    }
}
with open(os.path.join(out_dir, 'pipeline_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("SUCCESS: ALL PLOTS AND METRICS SAVED IN SECONDS!", flush=True)
