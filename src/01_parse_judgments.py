#! /usr/bin/python 
#-*-encoding:utf-8-*-
############################################
#
# Create time: 2025-10-20 19:22:22
# version 1.2 - 添加HTML解析支持
#
############################################
"""
01_parse_judgments.py
从判决书HTML文件进行信息抽取：使用BeautifulSoup解析HTML，然后用大模型提取特征。
输入：data/*.html（北大法宝格式的判决书HTML文件）
输出：JSON文件，格式为 {filename: {features}}
"""
import re, json, argparse, pathlib
from tqdm import tqdm
from bs4 import BeautifulSoup
from glm import chat

def parse_html_judgment(html_path):
    """
    解析北大法宝HTML格式的判决书
    返回：{"doc_id": ..., "text": ..., "court": ..., "case_number": ..., "year": ...}
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 提取文档ID（从文件名）
    doc_id = html_path.stem
    
    # 提取判决书正文
    fulltext_div = soup.find("div", {"id": "divFullText", "class": "fulltext"})
    if fulltext_div:
        # 去除HTML标签，保留文本
        text = fulltext_div.get_text(separator="\n", strip=True)
    else:
        text = ""
    
    # 提取元数据
    metadata = {}
    fields_div = soup.find("div", {"class": "fields"})
    
    if fields_div:
        # 提取案号
        case_number_elem = fields_div.find("div", {"class": "box"}, string=lambda s: s and "案 号：" in str(s))
        if case_number_elem:
            case_number_text = case_number_elem.get_text(strip=True)
            metadata["case_number"] = case_number_text.replace("案 号：", "").strip()
        
        # 提取审理法院
        for li in fields_div.find_all("li"):
            strong = li.find("strong")
            if strong:
                label = strong.get_text(strip=True)
                if "审理法院" in label:
                    links = li.find_all("a")
                    if links:
                        metadata["court"] = links[0].get_text(strip=True)
                elif "审结日期" in label:
                    box = li.find("div", {"class": "box"})
                    if box:
                        date_text = box.get_text(strip=True).replace("审结日期：", "").strip()
                        metadata["date"] = date_text
                        # 提取年份
                        year_match = re.search(r"(\d{4})", date_text)
                        if year_match:
                            metadata["year"] = int(year_match.group(1))
                elif "审理法官" in label or "审　判　长" in label:
                    judges = [a.get_text(strip=True) for a in li.find_all("a")]
                    if judges:
                        metadata["judges"] = ", ".join(judges)
    
    return {
        "doc_id": doc_id,
        "text": text,
        "court": metadata.get("court"),
        "case_number": metadata.get("case_number"),
        "year": metadata.get("year"),
        "date": metadata.get("date"),
        "judges": metadata.get("judges")
    }

def extract_features_with_llm(text, doc_id="", court="", judges="", year=None):
    """
    使用大模型提取判决书特征
    """
    prompt = f"""你是一名刑法量刑研究员，需要根据判决书文本提取关键特征，并对案件的关键要素进行量化评分。

【判决书内容】
{text[:8000]}

---

【任务一：基本特征提取】

请提取以下信息（严格按照JSON格式返回）：

{{
  "province": "法院所在省份（如：安徽省、北京市等）",
  "presiding_judge": "审判长姓名",
  "trial_year": 判决年份（整数）,
  "num_defendants": 被告人数量（整数，统计本案中被告人的总人数）,
  "total_bribe_amount": 受贿总金额（单位：元，整数）,
  "prior_party_or_admin_discipline": 是否曾因贪污、受贿、挪用公款受过党纪、行政处分（布尔值）,
  "prior_intent_crime_record": 是否曾因故意犯罪受过刑事追究（布尔值）,
  "used_for_illegal_activities": 赃款赃物是否用于非法活动（布尔值）,
  "refused_to_disclose_or_recover": 是否拒不交代去向或不配合追缴致无法追缴（布尔值）,
  "bad_impact_or_serious_consequence": 是否造成恶劣影响或其他严重后果（布尔值）,
  "repeated_solicitation": 是否多次索贿（布尔值）,
  "caused_public_loss": 是否为他人谋取不正当利益致公共/国家/人民利益遭受损失（布尔值）,
  "sought_promotion_adjustment": 是否为他人谋取职务提拔、调整（布尔值）,
  "sentence_months": 判处有期徒刑的月数（整数，如判处3年则为36）,
  "has_aggravating_circumstances": 是否有加重情节（布尔值）,
  "has_mitigating_circumstances": 是否有减轻情节（布尔值，包括从轻、减轻处罚情节）
}}

注意：
1. 如果某个信息在判决书中未明确提及，数字字段返回null，布尔字段返回false
2. 被告人数量指本案中所有被告人的人数（从"被告人XXX"等字样统计）
3. 受贿金额请提取总金额，单位转换为元（如56.5万元转换为565000）
4. 刑期如果是"三年"请转换为月数36
5. 加重情节包括：索贿、多次受贿、为他人谋取不正当利益造成损失等
6. 减轻情节包括：自首、坦白、认罪认罚、退赃、立功等

---

【任务二：统一量表评分】

请根据以下统一量表对案件的四个维度进行评分（0-5分）：

① 社会危害性（harm_score）：
| 等级 | 定义 |
|------|------|
| 0 | 无实质危害（轻微损失或未遂） |
| 1 | 较轻：金额 <1 万或后果轻微 |
| 2 | 一般：1–3 万，有轻微不良影响 |
| 3 | 较重：3–20 万或造成一定社会损害 |
| 4 | 重大：20–200 万或明显公共损失 |
| 5 | 特别重大：>200 万、影响恶劣或多人受害 |

② 人身危险性（risk_score）：
| 等级 | 定义 |
|------|------|
| 0 | 无再犯风险（偶发行为） |
| 1 | 较低：无前科，有悔罪表现 |
| 2 | 一般：轻度再犯风险、部分反复行为 |
| 3 | 较高：有前科或组织化特征 |
| 4 | 高：累犯、惯犯或明显职业化倾向 |
| 5 | 极高：反复多次、结伙、长期从事犯罪活动 |

③ 减轻情节（mitigating_score）：
| 等级 | 定义 |
|------|------|
| 0 | 无减轻因素 |
| 1 | 有轻微减轻情节（如认罪态度好） |
| 2 | 一般减轻（主动赔偿、部分退赃） |
| 3 | 明显减轻（自首或全面退赃、取得谅解） |
| 4 | 重大减轻（立功或特别积极改过） |
| 5 | 特别显著减轻（重大立功、检举他人、社会贡献大） |

④ 加重情节（aggravating_score）：
| 等级 | 定义 |
|------|------|
| 0 | 无加重情节 |
| 1 | 较轻（有轻度不良情节，如多次受贿但金额低） |
| 2 | 一般（有组织、预谋、受贿对象特殊） |
| 3 | 较重（主犯、关键职位、影响恶劣） |
| 4 | 重大（多次索贿、造成公共损失） |
| 5 | 特别重大（长期职业化、社会影响极坏） |

---

【最终输出格式】

请将任务一和任务二的结果合并，严格输出 JSON（不包含多余文本）：

{{
  "province": "...",
  "presiding_judge": "...",
  "trial_year": ...,
  "num_defendants": ...,
  "total_bribe_amount": ...,
  "prior_party_or_admin_discipline": ...,
  "prior_intent_crime_record": ...,
  "used_for_illegal_activities": ...,
  "refused_to_disclose_or_recover": ...,
  "bad_impact_or_serious_consequence": ...,
  "repeated_solicitation": ...,
  "caused_public_loss": ...,
  "sought_promotion_adjustment": ...,
  "sentence_months": ...,
  "has_aggravating_circumstances": ...,
  "has_mitigating_circumstances": ...,
  "harm_score": x,
  "risk_score": x,
  "mitigating_score": x,
  "aggravating_score": x,
  "explanations": {{
    "harm": "简要说明依据",
    "risk": "简要说明依据",
    "mitigating": "简要说明依据",
    "aggravating": "简要说明依据"
  }}
}}

请勿生成其他自然语言说明，只返回JSON。
"""
    
    try:
        # 调用大模型
        response = chat(prompt, model_name="glm-4-plus")
        result = json.loads(response)
        
        # 验证和补充字段
        extracted = {
            "doc_id": doc_id,
            "court": court,
            "judges": judges,
            "province": result.get("province"),
            "presiding_judge": result.get("presiding_judge"),
            "trial_year": result.get("trial_year") or year,
            "num_defendants": result.get("num_defendants"),
            "total_bribe_amount": result.get("total_bribe_amount"),
            "prior_party_or_admin_discipline": result.get("prior_party_or_admin_discipline", False),
            "prior_intent_crime_record": result.get("prior_intent_crime_record", False),
            "used_for_illegal_activities": result.get("used_for_illegal_activities", False),
            "refused_to_disclose_or_recover": result.get("refused_to_disclose_or_recover", False),
            "bad_impact_or_serious_consequence": result.get("bad_impact_or_serious_consequence", False),
            "repeated_solicitation": result.get("repeated_solicitation", False),
            "caused_public_loss": result.get("caused_public_loss", False),
            "sought_promotion_adjustment": result.get("sought_promotion_adjustment", False),
            "sentence_months": result.get("sentence_months"),
            "has_aggravating_circumstances": result.get("has_aggravating_circumstances", False),
            "has_mitigating_circumstances": result.get("has_mitigating_circumstances", False),
            # 统一量表评分
            "harm_score": result.get("harm_score"),
            "risk_score": result.get("risk_score"),
            "mitigating_score": result.get("mitigating_score"),
            "aggravating_score": result.get("aggravating_score"),
            "explanations": result.get("explanations", {})
        }
        
        return extracted
        
    except Exception as e:
        print(f"\nError extracting features for {doc_id}: {e}")
        # 返回空值的默认结果
        return {
            "doc_id": doc_id,
            "court": court,
            "judges": judges,
            "province": None,
            "presiding_judge": None,
            "trial_year": year,
            "num_defendants": None,
            "total_bribe_amount": None,
            "prior_party_or_admin_discipline": False,
            "prior_intent_crime_record": False,
            "used_for_illegal_activities": False,
            "refused_to_disclose_or_recover": False,
            "bad_impact_or_serious_consequence": False,
            "repeated_solicitation": False,
            "caused_public_loss": False,
            "sought_promotion_adjustment": False,
            "sentence_months": None,
            "has_aggravating_circumstances": False,
            "has_mitigating_circumstances": False,
            # 统一量表评分（默认值）
            "harm_score": None,
            "risk_score": None,
            "mitigating_score": None,
            "aggravating_score": None,
            "explanations": {}
        }

def main():
    ap = argparse.ArgumentParser(description="解析北大法宝HTML判决书并使用大模型提取特征")
    ap.add_argument("--raw", required=True, help="原始判决书HTML目录或单个HTML文件")
    ap.add_argument("--out", required=True, help="输出JSON文件路径")
    ap.add_argument("--limit", type=int, default=None, help="限制处理的文件数量（用于测试）")
    args = ap.parse_args()
    
    # 获取所有HTML文件
    path = pathlib.Path(args.raw)
    if path.is_dir():
        files = list(path.glob("*.html"))
    elif path.is_file() and path.suffix == ".html":
        files = [path]
    else:
        raise ValueError(f"路径必须是目录或HTML文件: {args.raw}")
    
    # 限制文件数量（用于测试）
    if args.limit:
        files = files[:args.limit]
    
    print(f"Found {len(files)} HTML files to process")
    
    # 解析HTML文件
    print("\n[1/2] Parsing HTML files...")
    rows = []
    filename_map = {}  # 保存文件名映射
    for f in tqdm(files, desc="Parsing HTML"):
        try:
            row_data = parse_html_judgment(f)
            rows.append(row_data)
            filename_map[row_data["doc_id"]] = f.name  # 保存原始文件名
        except Exception as e:
            print(f"\nError parsing {f.name}: {e}")
    
    print(f"\nSuccessfully parsed {len(rows)} documents")
    
    # 使用大模型提取特征
    print("\n[2/2] Extracting features with LLM...")
    results_dict = {}  # 使用字典存储，key为文件名
    for row in tqdm(rows, desc="LLM extraction"):
        try:
            features = extract_features_with_llm(
                text=row["text"],
                doc_id=row["doc_id"],
                court=row.get("court"),
                judges=row.get("judges"),
                year=row.get("year")
            )
            # 使用文件名作为key，移除doc_id字段（因为已经是key了）
            filename = filename_map.get(row["doc_id"], row["doc_id"])
            features_copy = features.copy()
            features_copy.pop("doc_id", None)  # 移除doc_id，因为已经作为key
            results_dict[filename] = features_copy
        except Exception as e:
            print(f"\nError processing {row.get('doc_id')}: {e}")
    
    # 保存为JSON
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Successfully saved: {args.out}")
    print(f"  Total files: {len(results_dict)}")
    print(f"  Format: JSON (filename -> features)")
    print(f"\n前3个文件的特征:")
    for i, (filename, features) in enumerate(list(results_dict.items())[:3]):
        print(f"\n  [{i+1}] {filename}")
        for key, value in list(features.items())[:5]:  # 只显示前5个字段
            print(f"      {key}: {value}")
        if len(features) > 5:
            print(f"      ... (共{len(features)}个字段)")

if __name__ == "__main__":
    main()

