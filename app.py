import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
TICKETS=DATA/"tickets.csv"
AGENTS=DATA/"agents.csv"
PRODUCTS=DATA/"products.csv"

st.set_page_config(page_title="Vireo Audio Support", layout="wide")

@st.cache_data
def load():
    df=pd.read_csv(TICKETS)
    agents=pd.read_csv(AGENTS)
    products=pd.read_csv(PRODUCTS)
    for c in ["created_at","first_response_at","resolved_at"]:
        df[c]=pd.to_datetime(df[c],errors="coerce")
    df["response_min"]=(df.first_response_at-df.created_at).dt.total_seconds()/60
    df["handle_min"]=(df.resolved_at-df.first_response_at).dt.total_seconds()/60
    df=df.merge(agents[["agent_id","name","site","team","shift","tier"]],on="agent_id",how="left")
    df["sla_target_min"]=df.channel.map({"chat":15,"voice":120,"social":240,"email":480})
    df["sla_breach"]=df.response_min>df.sla_target_min
    df["attendance"]=df.status.isin(["resolved","closed"])
    df=df.merge(products[["sku","unit_cost_inr","product_name"]],left_on="product_sku",right_on="sku",how="left")
    df["replacement_cost"]=np.where(df.replacement_issued.eq("Y"),df.unit_cost_inr+340,0)
    return df

df=load()
st.title("Vireo Audio — Support Performance")
st.caption("Dashboard-first, policy-aligned analytics. Tier-2 is separated; legacy negative handle intervals are not imputed.")

# KPI
csat=df.csat_score.dropna()
valid=df.handle_min>=0
cols=st.columns(6)
vals=[
    ("Tickets",f"{len(df):,}"),
    ("Mean CSAT",f"{csat.mean():.2f}/5"),
    ("CSAT response",f"{df.csat_score.notna().mean()*100:.1f}%"),
    ("SLA breach",f"{df.sla_breach.mean()*100:.1f}%"),
    ("Replacements",f"{(df.replacement_issued=='Y').sum():,}"),
    ("Replacement cost",f"₹{df.replacement_cost.sum()/1e5:.2f}L"),
]
for c,(a,b) in zip(cols,vals):
    c.metric(a,b)

st.warning(f"Data-quality: {(df.handle_min<0).sum():,} legacy tickets have negative handle intervals. They are excluded from valid handle-time metrics; no timestamp correction is fabricated.")

tab1,tab2,tab3,tab4=st.tabs(["Overview","Agent KPIs","Replacement","AI Text Model"])

with tab1:
    monthly=df.groupby(df.created_at.dt.to_period("M")).agg(
        tickets=("ticket_id","size"),
        replacements=("replacement_issued",lambda s:(s=="Y").sum())
    ).reset_index()
    monthly["month"]=monthly.created_at.astype(str)
    fig=px.line(monthly,x="month",y=["tickets","replacements"],markers=True,title="Monthly volume and replacements")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(df.groupby("channel").agg(tickets=("ticket_id","size"),sla_breach_rate=("sla_breach","mean"),csat=("csat_score","mean")).reset_index().assign(sla_breach_rate=lambda x:(x.sla_breach_rate*100).round(1),csat=lambda x:x.csat.round(2)),use_container_width=True)

with tab2:
    min_resp=st.slider("Minimum CSAT responses for bottom-ten flag",20,100,50,5)
    g=df.groupby(["agent_id","name","site","team","tier"]).agg(
        tickets=("ticket_id","size"),
        csat_responses=("csat_score",lambda s:s.notna().sum()),
        csat=("csat_score","mean"),
        sla_breach_rate=("sla_breach","mean"),
        valid_handle=("handle_min",lambda s:(s>=0).sum()),
        median_handle_min=("handle_min",lambda s:s[s>=0].median()),
        replacements=("replacement_issued",lambda s:(s=="Y").sum())
    ).reset_index()
    tier1=g[g.tier==1].copy()
    bottom=tier1[tier1.csat_responses>=min_resp].sort_values(["csat","sla_breach_rate"]).head(10)
    st.subheader("Tier-1 bottom-ten investigation queue")
    st.dataframe(bottom.assign(csat=lambda x:x.csat.round(2),sla_breach_rate=lambda x:(x.sla_breach_rate*100).round(1),median_handle_min=lambda x:x.median_handle_min.round(1)),use_container_width=True)
    st.caption("This is a coaching/investigation queue, not a disciplinary ranking. Tier-2 is intentionally excluded.")
    st.subheader("All agent KPIs")
    st.dataframe(g.sort_values(["tier","csat"]),use_container_width=True)

with tab3:
    r=df[df.replacement_issued=="Y"].copy()
    prod=r.groupby("product_name").agg(replacements=("ticket_id","size"),planning_cost=("replacement_cost","sum")).reset_index().sort_values("replacements",ascending=False).head(10)
    st.plotly_chart(px.bar(prod,x="replacements",y="product_name",orientation="h",title="Top products by replacement volume"),use_container_width=True)
    st.write(f"**15% planning target:** approximately ₹{df.replacement_cost.sum()*0.15:,.0f} replacement planning cost avoided on the current baseline. This is a target to validate, not a causal finding.")
    st.dataframe(prod,use_container_width=True)

with tab4:
    st.write("Local AI-assisted category suggestion using customer_message only. No paid LLM calls.")
    @st.cache_resource
    def train_model():
        text=df.customer_message.fillna("")
        y=df.category.fillna("Other")
        model=Pipeline([
            ("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000,sublinear_tf=True)),
            ("lr",LogisticRegression(max_iter=1000,C=3,class_weight="balanced"))
        ])
        model.fit(text,y)
        return model
    model=train_model()
    q=st.text_area("Paste a ticket message for a category suggestion", "My left earbud is not charging at all")
    if q:
        proba=model.predict_proba([q])[0]
        idx=int(np.argmax(proba))
        st.metric("Suggested category",model.classes_[idx],f"{proba[idx]*100:.1f}% confidence")
        top=np.argsort(proba)[::-1][:3]
        st.write(pd.DataFrame({"category":model.classes_[top],"confidence":proba[top]}).assign(confidence=lambda x:(x.confidence*100).round(1)))
    st.info("Evaluation from the build: 2,350-ticket stratified holdout, 77.4% accuracy. At confidence ≥0.60, coverage was 77.7% with 80.8% accuracy. Low-confidence cases should remain human-reviewed.")
