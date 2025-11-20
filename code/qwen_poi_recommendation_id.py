import json
import time
from openai import OpenAI

def load_data(file_path):
    """
    加载JSON格式的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def evaluate_recommendation_with_ids():
    """
    使用Qwen API进行POI推荐并评估（使用ID格式数据）
    """
    # 加载训练和测试数据（使用ID格式）
    train_data = load_data('datasets/NYC/train_id.json')
    test_data = load_data('datasets/NYC/test_id.json')
    
    # 获取OpenAI客户端（需要配置Qwen API密钥）
    client = OpenAI(
        api_key="sk-36ce00113a054e9d830c6945c3c28842",  # 替换为实际的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    print(f"开始评估ID格式数据，总测试样本数: {total_predictions}")
    
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
                    {"role": "system", "content": "你是一个专业的POI（兴趣点）推荐助手，根据用户的历史访问记录预测下一个可能访问的POI。请直接输出预测的POI ID，格式如 <数字ID>。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,  # 限制输出长度
                temperature=0.1,  # 降低随机性，使输出更确定
                extra_body={"enable_thinking": False}
            )
            
            predicted_output = response.choices[0].message.content.strip()
            
            # 检查预测是否正确
            is_correct = check_id_prediction_accuracy(predicted_output, true_output)
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
    print(f"\nID格式数据评估完成!")
    print(f"总样本数: {total_predictions}")
    print(f"正确预测数: {correct_predictions}")
    print(f"ACC@1: {final_acc:.4f}")
    
    return final_acc

def extract_id_codes(text):
    """
    从文本中提取ID格式的POI代码，返回所有匹配的ID代码列表
    ID格式为 <数字> 如 <1796>
    """
    import re
    # 匹配ID格式的正则表达式
    pattern = r'<\d+>'
    matches = re.findall(pattern, text)
    
    return matches

def check_id_prediction_accuracy(predicted, actual):
    """
    检查ID格式预测结果的准确性 (ACC@1)
    ACC@1 只关心第一个预测的POI ID是否正确
    """
    # 移除多余的空格和换行符
    predicted = predicted.strip()
    actual = actual.strip()
    
    # 提取预测文本中的所有ID格式POI代码
    predicted_id_codes = extract_id_codes(predicted)
    
    # ACC@1: 检查第一个预测的POI ID是否与实际POI ID匹配
    if len(predicted_id_codes) > 0:
        first_predicted = predicted_id_codes[0]
        return first_predicted == actual
    else:
        # 如果没有找到ID格式的POI代码，则预测不正确
        return False

def main():
    """
    主函数
    """
    print("开始使用Qwen API进行ID格式POI推荐评估...")
    
    # 执行评估
    acc_score = evaluate_recommendation_with_ids()
    
    print(f"\n最终ID格式数据ACC@1得分: {acc_score:.4f}")

if __name__ == "__main__":
    main()