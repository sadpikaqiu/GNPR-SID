import json
import random
import time
from openai import OpenAI
from collections import Counter

def load_data(file_path):
    """
    加载JSON格式的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_recommendation():
    """
    使用Qwen API进行POI推荐并评估
    """
    # 加载训练和测试数据
    train_data = load_data('datasets/NYC/train_codebook.json')
    test_data = load_data('datasets/NYC/test_codebook.json')
    
    # 获取OpenAI客户端（需要配置Qwen API密钥）
    client = OpenAI(
        api_key="MY-API",  # 替换为实际的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    print(f"开始评估，总测试样本数: {total_predictions}")
    
    for i, sample in enumerate(test_data):
        instruction = sample['instruction']
        input_history = sample['input']
        true_output = sample['output']
        
        # 构建提示
        prompt = f"{instruction}\n\n{input_history}"
        
        try:
            # 调用Qwen API进行预测
            response = client.chat.completions.create(
                model="qwen3-8b",  # 可以选择其他模型如qwen-plus或qwen-turbo
                messages=[
                    {"role": "system", "content": "你是一个专业的POI（兴趣点）推荐助手，根据用户的历史访问记录预测下一个可能访问的POI。请直接输出预测的POI语义ID，格式如 <a_x><b_y><c_z>。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,  # 限制输出长度
                temperature=0.1,  # 降低随机性，使输出更确定
                extra_body={"enable_thinking": False}
            )
            
            predicted_output = response.choices[0].message.content.strip()
            
            # 检查预测是否正确
            is_correct = check_prediction_accuracy(predicted_output, true_output)
            if is_correct:
                correct_predictions += 1
                
            print(f"样本 {i+1}/{total_predictions}:")
            print(f"  输入: {input_history}")
            print(f"  真实输出: {true_output}")
            print(f"  预测输出: {predicted_output}")
            print(f"  是否正确: {is_correct}")
            print(f"  当前ACC@1: {correct_predictions}/{i+1} = {correct_predictions/(i+1):.4f}")
            print("-" * 50)
            
            # 添加延迟以避免API调用过于频繁
            time.sleep(0.5)
            
        except Exception as e:
            print(f"处理样本 {i+1} 时出错: {str(e)}")
            continue
    
    # 计算最终的ACC@1指标
    final_acc = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\n评估完成!")
    print(f"总样本数: {total_predictions}")
    print(f"正确预测数: {correct_predictions}")
    print(f"ACC@1: {final_acc:.4f}")
    
    return final_acc

def extract_poi_codes(text):
    """
    从文本中提取POI代码，返回所有匹配的POI代码列表
    POI代码格式为 <a_x><b_y><c_z> 或 <a_x><b_y><c_z><d_w>
    """
    import re
    # 匹配POI代码的正则表达式
    pattern = r'<[a-z]_\d+><[a-z]_\d+><[a-z]_\d+>(<[a-z]_\d+>)?'
    matches = re.findall(pattern, text)

    # re.findall只返回捕获组，所以我们需要重新构建完整匹配
    full_matches = re.findall(pattern, text)
    # 但我们需要完整匹配，所以使用finditer
    poi_codes = []
    for match in re.finditer(pattern, text):
        poi_codes.append(match.group(0))

    return poi_codes

def check_prediction_accuracy(predicted, actual):
    """
    检查预测结果的准确性 (ACC@1)
    ACC@1 只关心第一个预测的POI是否正确
    """
    # 移除多余的空格和换行符
    predicted = predicted.strip()
    actual = actual.strip()

    # 提取预测文本中的所有POI代码
    predicted_poi_codes = extract_poi_codes(predicted)

    # ACC@1: 检查第一个预测的POI是否与实际POI匹配
    if len(predicted_poi_codes) > 0:
        first_predicted = predicted_poi_codes[0]
        return first_predicted == actual
    else:
        # 如果没有找到POI代码，则预测不正确
        return False

def main():
    """
    主函数
    """
    print("开始使用Qwen API进行POI推荐评估...")
    
    # 执行评估
    acc_score = evaluate_recommendation()
    
    print(f"\n最终ACC@1得分: {acc_score:.4f}")

if __name__ == "__main__":
    main()