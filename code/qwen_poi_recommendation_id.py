import json
import time
import re
from openai import OpenAI

def load_data(file_path):
    """
    加载JSON格式的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_user_history_map(history_data):
    """
    创建用户ID到历史轨迹的映射
    """
    user_history_map = {}
    for item in history_data:
        # 从input字段中提取用户ID (格式如 "User_1 visited:...")
        input_text = item['input']
        user_match = re.search(r'User_(\d+)', input_text)
        if user_match:
            user_id = int(user_match.group(1))
            user_history_map[user_id] = input_text
    return user_history_map

def evaluate_recommendation_with_ids(use_history=True):
    """
    使用Qwen API进行POI推荐并评估（使用ID格式数据）

    参数:
    use_history (bool): 是否使用用户的历史轨迹作为参考，默认为True
    """
    # 加载测试数据
    test_data = load_data('datasets/NYC/data/test_id.json')

    # 如果需要使用历史轨迹，则加载历史数据
    user_history_map = {}
    if use_history:
        history_data = load_data('datasets/NYC/data/history_id.json')
        user_history_map = create_user_history_map(history_data)

    # 获取OpenAI客户端（需要配置Qwen API密钥）
    client = OpenAI(
        api_key="sk-36ce00113a054e9d830c6945c3c28842",  # 替换为实际的API密钥
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 初始化acc1, acc5, acc10的计数器
    correct_predictions_at_1 = 0
    correct_predictions_at_5 = 0
    correct_predictions_at_10 = 0
    total_predictions = len(test_data)

    print(f"开始评估ID格式数据，总测试样本数: {total_predictions}")
    print(f"使用历史轨迹: {use_history}")

    for i, sample in enumerate(test_data):
        instruction = sample['instruction']
        input_sequence = sample['input']
        true_output = sample['output']

        # 从当前序列中提取用户ID
        user_match = re.search(r'User_(\d+)', input_sequence)
        user_id = None
        if user_match:
            user_id = int(user_match.group(1))

        # 获取用户的历史轨迹（如果启用）
        user_history = ""
        if use_history and user_id and user_id in user_history_map:
            user_history = f"用户{user_id}的完整历史轨迹: {user_history_map[user_id]}\n\n"
        elif use_history and user_id and user_id not in user_history_map:
            print(f"警告: 未找到用户 {user_id} 的历史轨迹")

        # 构建提示
        if use_history and user_history:
            # 包含用户历史轨迹和当前序列
            prompt = f"{instruction}\n\n{user_history}当前序列: {input_sequence}"
        else:
            # 仅当前序列
            prompt = f"{instruction}\n\n{input_sequence}"

        try:
            # 调用Qwen API进行预测
            system_content = "你是一个专业的POI（兴趣点）推荐助手，"
            if use_history:
                system_content += "结合用户的完整历史访问记录和当前访问序列，预测用户下一个可能访问的POI。"
            else:
                system_content += "根据用户的当前访问序列，预测用户下一个可能访问的POI。"
            system_content += "请直接输出预测的POI ID，最多输出10个最可能的POI，按优先级排序，格式如 <数字ID> <数字ID> <数字ID>...。"

            response = client.chat.completions.create(
                model="qwen3-8b",  # 可以选择其他模型如qwen-plus或qwen-turbo
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,  # 限制输出长度
                temperature=0.1,  # 降低随机性，使输出更确定
                extra_body={"enable_thinking": False}
            )

            predicted_output = response.choices[0].message.content.strip()

            # 检查预测是否正确 (Top-1, Top-5, Top-10)
            is_correct_at_1 = check_id_prediction_accuracy(predicted_output, true_output, k=1)
            is_correct_at_5 = check_id_prediction_accuracy(predicted_output, true_output, k=5)
            is_correct_at_10 = check_id_prediction_accuracy(predicted_output, true_output, k=10)

            if is_correct_at_1:
                correct_predictions_at_1 += 1
            if is_correct_at_5:
                correct_predictions_at_5 += 1
            if is_correct_at_10:
                correct_predictions_at_10 += 1

            print(f"样本 {i+1}/{total_predictions}:")
            print(f"  用户ID: {user_id}")
            print(f"  使用历史轨迹: {use_history}")
            print(f"  输入: {input_sequence}")
            if use_history and user_history:
                print(f"  历史轨迹: {user_history_map[user_id][:100]}...")  # 只打印前100个字符作为示例
            print(f"  真实输出: {true_output}")
            print(f"  预测输出: {predicted_output}")
            print(f"  Top-1是否正确: {is_correct_at_1}, Top-5是否正确: {is_correct_at_5}, Top-10是否正确: {is_correct_at_10}")
            print(f"  当前ACC@1: {correct_predictions_at_1}/{i+1} = {correct_predictions_at_1/(i+1):.4f}")
            print(f"  当前ACC@5: {correct_predictions_at_5}/{i+1} = {correct_predictions_at_5/(i+1):.4f}")
            print(f"  当前ACC@10: {correct_predictions_at_10}/{i+1} = {correct_predictions_at_10/(i+1):.4f}")
            print("-" * 50)

            # 添加延迟以避免API调用过于频繁
            time.sleep(0.5)

        except Exception as e:
            print(f"处理样本 {i+1} 时出错: {str(e)}")
            continue

    # 计算最终的ACC@1, ACC@5, ACC@10指标
    final_acc_at_1 = correct_predictions_at_1 / total_predictions if total_predictions > 0 else 0
    final_acc_at_5 = correct_predictions_at_5 / total_predictions if total_predictions > 0 else 0
    final_acc_at_10 = correct_predictions_at_10 / total_predictions if total_predictions > 0 else 0

    print(f"\nID格式数据评估完成!")
    print(f"总样本数: {total_predictions}")
    print(f"使用历史轨迹: {use_history}")
    print(f"ACC@1: {final_acc_at_1:.4f} ({correct_predictions_at_1}/{total_predictions})")
    print(f"ACC@5: {final_acc_at_5:.4f} ({correct_predictions_at_5}/{total_predictions})")
    print(f"ACC@10: {final_acc_at_10:.4f} ({correct_predictions_at_10}/{total_predictions})")

    return final_acc_at_1, final_acc_at_5, final_acc_at_10

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

def check_id_prediction_accuracy(predicted, actual, k=1):
    """
    检查ID格式预测结果的准确性 (Top-K)
    k: 检查前k个预测中是否包含真实POI ID
    """
    # 移除多余的空格和换行符
    predicted = predicted.strip()
    actual = actual.strip()

    # 提取预测文本中的所有ID格式POI代码
    predicted_id_codes = extract_id_codes(predicted)

    # 检查前k个预测中是否包含实际值
    top_k_predictions = predicted_id_codes[:k] if len(predicted_id_codes) >= k else predicted_id_codes

    # 检查真实值是否在前k个预测中
    return actual in top_k_predictions

def main():
    """
    主函数
    """
    # 默认使用历史轨迹，也可以通过参数控制
    # 例如：evaluate_recommendation_with_ids(use_history=True) 或 evaluate_recommendation_with_ids(use_history=False)
    use_history = False  # 可以在这里手动更改设置

    mode_text = "结合用户历史轨迹" if use_history else "仅使用当前序列"
    print(f"开始使用Qwen API进行ID格式POI推荐评估（{mode_text}）...")

    # 执行评估
    acc_score_at_1, acc_score_at_5, acc_score_at_10 = evaluate_recommendation_with_ids(use_history=use_history)

    print(f"\n最终ID格式数据评估结果:")
    print(f"ACC@1: {acc_score_at_1:.4f}")
    print(f"ACC@5: {acc_score_at_5:.4f}")
    print(f"ACC@10: {acc_score_at_10:.4f}")

if __name__ == "__main__":
    main()