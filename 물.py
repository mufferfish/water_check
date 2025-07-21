import streamlit as st

st.set_page_config(page_title="물 위험도 평가기", page_icon="💧")

st.title("💧 이 물, 얼마나 위험할까요?")
st.markdown("간단한 설문에 답하면 물의 **위험 수준**을 판단해드립니다.")

# 👉 설문 입력
ph_q = st.radio("1. 물에 거품이 많이 생기나요?", ["O", "X"])
rust_q = st.radio("2. 물이 붉거나 갈색인가요?", ["O", "X"])
solids_q = st.slider("3. 물에 이물질이 보이나요? (탁한 정도)", 0, 50000, 15000)
chlorine_q = st.radio("4. 소독약(염소) 냄새가 많이 나나요?", ["O", "X"])
metallic_q = st.radio("5. 금속 맛이 느껴지나요?", ["O", "X"])
smell_q = st.radio("6. 냄새가 나나요?", ["O", "X"])
trihalo_q = st.radio("7. 오래된 물 맛이 나거나 불쾌한 맛이 나나요?", ["O", "X"])

# 위험도 판단 함수
def evaluate_water_risk(ph_q, rust_q, chlorine_q, metallic_q, smell_q, trihalo_q, solids_q):
    reasons = []
    if ph_q == "O":
        reasons.append("pH 불균형 가능성 (거품 발생)")
    if rust_q == "O":
        reasons.append("산화철 또는 녹이 섞인 물")
    if chlorine_q == "O":
        reasons.append("잔류 염소 농도 과다 가능성")
    if metallic_q == "O":
        reasons.append("중금속류 또는 배관 오염 가능성")
    if smell_q == "O":
        reasons.append("미생물 또는 유기물 오염 가능성")
    if trihalo_q == "O":
        reasons.append("트리할로메탄(발암 물질) 가능성")
    if solids_q > 30000:
        reasons.append("탁도 과다 (이물질 농도 높음)")

    risk_count = len(reasons)

    if risk_count >= 3:
        risk_level = "🔴 높은 위험"
        summary = "여러 문제점이 발견되어 **음용을 피해야 합니다.**"
    elif 1 <= risk_count <= 2:
        risk_level = "🟠 주의 필요"
        summary = "몇 가지 우려 사항이 있어 **음용 전 주의가 필요합니다.**"
    else:
        risk_level = "🟢 안전 추정"
        summary = "특별한 문제는 발견되지 않았습니다. **마셔도 괜찮아 보입니다.** 😊"

    return risk_level, summary, reasons

# 결과 출력
if st.button("🔍 물 위험도 평가하기"):
    level, summary, reasons = evaluate_water_risk(ph_q, rust_q, chlorine_q, metallic_q, smell_q, trihalo_q, solids_q)

    st.markdown("---")
    st.subheader(f"📊 위험도 평가 결과: {level}")
    st.write(summary)

    if reasons:
        st.markdown("**❗ 감지된 문제:**")
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.success("✔ 모든 항목이 양호합니다!")

    # 입력 요약
    st.markdown("📋 **입력 요약:**")
    st.write({
        "거품 여부": ph_q,
        "색깔 이상": rust_q,
        "이물질 양": solids_q,
        "염소 냄새": chlorine_q,
        "금속 맛": metallic_q,
        "냄새 여부": smell_q,
        "이상한 맛": trihalo_q
    })

st.markdown("---")
st.markdown("🔗 더 많은 AI 앱 만들기는 [gptonline.ai/ko](https://gptonline.ai/ko/)에서 확인해보세요!")
