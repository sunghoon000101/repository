import streamlit as st
import pandas as pd
from rdkit import Chem
from utils import (
    load_data, find_activity_cliffs, generate_hypothesis, draw_highlighted_pair,
    load_pretrained_model, smiles_to_descriptors, load_feature_list,
    propose_and_predict_analogs
)
import plotly.express as px
import os

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 기반 SAR/QSAR 분석 시스템",
    page_icon="🧪",
    layout="wide",
)

# --- 사이드바 UI ---
with st.sidebar:
    st.title("AI SAR/QSAR 분석 시스템")
    st.info("신약 개발을 위한 AI 기반 구조-활성 관계 분석 및 예측 솔루션입니다.")
    
    st.header("1. 데이터 입력")
    uploaded_file = st.file_uploader("SAR/QSAR 분석용 CSV 파일을 업로드하세요.", type="csv")
    use_sample_data = st.checkbox("샘플 데이터 사용", value=True)

    st.header("2. AI 모델 설정")
    target_name = st.text_input("분석 대상 타겟 단백질 (예: EGFR)", value="EGFR")
    
    llm_provider = st.selectbox("LLM 공급자 선택:", ("OpenAI", "Gemini"))
    
    api_key_placeholder = "OpenAI API 키 (sk-...)" if llm_provider == "OpenAI" else "Gemini API 키"
    api_key = st.text_input("API 키 입력:", type="password", placeholder=api_key_placeholder)

st.header("분석 결과 대시보드")

# --- 데이터 로딩 ---
df = None
if use_sample_data:
    # 로컬 경로에 맞게 수정
    sample_path = 'data/large_sar_data.csv'
    if os.path.exists(sample_path):
        df = load_data(sample_path)
    else:
        st.sidebar.error(f"샘플 데이터 파일 '{sample_path}'를 찾을 수 없습니다.")
elif uploaded_file:
    df = load_data(uploaded_file)

# --- 탭 구성 ---
if df is not None:
    tab1, tab2 = st.tabs(["SAR 분석 (Activity Cliff)", "AI 기반 분자 최적화 (QSAR)"])

    # ==================== SAR 분석 탭 ====================


    with tab1:
        st.subheader("🎯 Activity Cliff 자동 분석 리포트")

        # --- SAR 탭 데이터 시각화 ---
        with st.expander("현재 데이터 활성도 분포 보기"):
            plot_df = df.copy()
            if 'pKi' in plot_df.columns:
                plot_df['pKi'] = pd.to_numeric(plot_df['pKi'], errors='coerce')
                plot_df.dropna(subset=['pKi'], inplace=True)
                
                if not plot_df.empty:

                    display_df = plot_df.drop(columns=['mol', 'fp', 'scaffold'], errors='ignore')
                    st.dataframe(display_df.head()) # RDKit 객체가 제거된 display_df를 표시
                    fig = px.histogram(plot_df, x='pKi', title='활성도(pKi) 분포', labels={'pKi': 'pKi 값'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("'pKi' 컬럼에 유효한 숫자 데이터가 없어 분포를 표시할 수 없습니다.")
            else:
                st.warning("'pKi' 컬럼을 찾을 수 없어 분포를 표시할 수 없습니다.")

        st.subheader("🎯 Activity Cliff 자동 분석 리포트")
        
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider("유사도 임계값 (Tanimoto)", 0.5, 1.0, 0.8, 0.01)
        with col2:
            activity_diff_threshold = st.slider("활성도 차이 임계값 (ΔpKi)", 0.5, 5.0, 1.0, 0.1)

        if st.button("Activity Cliff 분석 시작", key='sar_analyze'):
            with st.spinner("Activity Cliff 분석 중..."):
                cliffs = find_activity_cliffs(df, similarity_threshold, activity_diff_threshold)
                st.session_state['cliffs'] = cliffs

        if 'cliffs' in st.session_state:
            cliffs = st.session_state['cliffs']
            if not cliffs:
                st.warning("설정된 조건에 맞는 Activity Cliff를 찾을 수 없습니다.")
            else:
                st.success(f"총 {len(cliffs)}개의 Activity Cliff를 찾았습니다. 분석할 쌍을 선택하세요.")
                
                cliff_options = [f"{i+1}. {c['mol_1']['ID']} vs {c['mol_2']['ID']} (ΔpKi: {c['activity_diff']:.2f})" for i, c in enumerate(cliffs)]
                selected_option = st.selectbox("분석할 Activity Cliff 선택:", cliff_options, key='cliff_select')
                
                if selected_option:
                    selected_index = cliff_options.index(selected_option)
                    selected_cliff = cliffs[selected_index]

                    mol1_info = selected_cliff['mol_1']
                    mol2_info = selected_cliff['mol_2']

                    st.markdown("---")
                    st.markdown(f"#### 선택된 Cliff: **{mol1_info['ID']}** vs **{mol2_info['ID']}**")
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tanimoto 유사도", f"{selected_cliff['similarity']:.3f}")
                    c2.metric("pKi 차이 (ΔpKi)", f"{selected_cliff['activity_diff']:.3f}")
                    
                    svg1, svg2 = draw_highlighted_pair(mol1_info['SMILES'], mol2_info['SMILES'])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**화합물 1: {mol1_info['ID']}** (pKi: {mol1_info['pKi']:.2f})")
                        if svg1:
                            st.image(svg1, use_column_width=True)
                        else:
                            st.warning("이미지를 생성할 수 없습니다.")
                    
                    with col2:
                        st.markdown(f"**화합물 2: {mol2_info['ID']}** (pKi: {mol2_info['pKi']:.2f})")
                        if svg2:
                            st.image(svg2, use_column_width=True)
                        else:
                            st.warning("이미지를 생성할 수 없습니다.")
                    
                    with st.spinner("AI가 참고 문헌을 검색하고 가설을 생성 중입니다..."):
                        hypothesis, source_info = generate_hypothesis(selected_cliff, target_name, api_key, llm_provider)
                    
                    st.markdown("---")
                    st.markdown("#### 🤖 AI-Generated Hypothesis")
                    st.markdown(hypothesis)

                    if source_info:
                        with st.expander("📚 참고 문헌 정보 (RAG 근거)"):
                            st.markdown(f"**- 제목:** {source_info['title']}")
                            st.markdown(f"**- 링크:** [PubMed]({source_info['link']})")
                            st.markdown(f"**- 초록:** {source_info['abstract']}")

    # ==================== QSAR 예측 탭 ====================
    with tab2:
        st.subheader("💡 AI 기반 분자 최적화 제안 (QSAR)")
        
        model_pipeline, model_msg = load_pretrained_model()
        features, features_msg = load_feature_list()

        if model_pipeline and features:
            st.text_input("최적화할 분자의 SMILES를 입력하세요:", key="base_smiles_qsar")
            
            if st.button("AI 최적화 제안 시작", key="qsar_predict"):
                if st.session_state.base_smiles_qsar:
                    with st.spinner("AI가 더 나은 분자를 설계하고 QSAR로 활성을 예측 중입니다..."):
                        proposals = propose_and_predict_analogs(
                            st.session_state.base_smiles_qsar,
                            model_pipeline,
                            features,
                            api_key,
                            llm_provider
                        )
                    
                    st.session_state['proposals'] = proposals
                else:
                    st.warning("SMILES 문자열을 입력해주세요.")

            if 'proposals' in st.session_state and st.session_state['proposals'] is not None:
                proposals = st.session_state['proposals']
                st.markdown("---")
                st.success("AI가 활성 개선이 예상되는 새로운 분자 3개를 제안했습니다.")

                base_features = smiles_to_descriptors(st.session_state.base_smiles_qsar, features)
                if base_features is not None:
                    base_pki = model_pipeline.predict(base_features.reshape(1, -1))[0]
                else:
                    base_pki = 0

                st.markdown("##### **기준 분자**")
                col1, col2 = st.columns([1, 2])
                with col1:
                    base_svg, _ = draw_highlighted_pair(st.session_state.base_smiles_qsar, st.session_state.base_smiles_qsar)
                    if base_svg:
                        st.markdown(base_svg, unsafe_allow_html=True)
                with col2:
                    st.metric("예상 pKi", f"{base_pki:.2f}")
                    st.caption(st.session_state.base_smiles_qsar)

                st.markdown("---")
                st.markdown("##### **AI 제안 분자**")
                for i, prop in enumerate(proposals):
                    col1_prop, col2_prop = st.columns([1, 2])
                    with col1_prop:
                        _, prop_svg = draw_highlighted_pair(st.session_state.base_smiles_qsar, prop['smiles'])
                        if prop_svg:
                            st.markdown(prop_svg, unsafe_allow_html=True)
                    with col2_prop:
                        st.metric(f"제안 {i+1} 예상 pKi", f"{prop['predicted_pki']:.2f}", delta=f"{prop['predicted_pki'] - base_pki:.2f}")
                        st.info(f"**변경 이유:** {prop['rationale']}")
                        st.caption(prop['smiles'])
                    st.markdown("---")
        else:
            st.error(f"{model_msg}\n{features_msg}")
else:
    st.info("사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 선택하여 분석을 시작하세요.")

