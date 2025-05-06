import json, os


def remove_newlines_in_text(input_file: str, output_file: str):
    """
    去除JSON文件中所有text字段的换行符
    参数：
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    count = 0
    try:
        # 读取原始文件
        with open(input_file, 'r', encoding = 'utf-8') as f:
            data = json.load(f)

        # 处理每个条目的text字段
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'text' in item:
                    item['text'] = item['text'].replace('\n', '')
                    item['text'] = item['text'].replace('\t', '')
                    item['text'] = item['text'].replace('\\', '')
                    item['text'] = item['text'].replace('\r', '')
                    item['text'] = item['text'].replace('\"', '')
                    item['text'] = item['text'].replace('\b', '')
                    count += 1
        elif isinstance(data, dict):
            if 'text' in data:
                data['text'] = data['text'].replace('\n', '')
                data['text'] = data['text'].replace('\t', '')
                data['text'] = data['text'].replace('\\', '')
                data['text'] = data['text'].replace('\r', '')
                data['text'] = data['text'].replace('\"', '')
                count += 1

        # 保存处理后的文件
        with open(output_file, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, ensure_ascii = False, indent = 2)

        print(f"处理完成，已保存到: {output_file} \n处理了{count}条数据")

    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
    except json.JSONDecodeError:
        print("错误：文件格式不符合JSON规范")
    except Exception as e:
        print(f"未知错误: {str(e)}")


if __name__ == "__main__":
    folder_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # 处理训练集
    input_json = os.path.join(folder_path,"data/train.json")  # 原始文件路径
    output_json = os.path.join(folder_path,"data/train_dealed.json")  # 输出文件路径
    remove_newlines_in_text(input_json, output_json)

    # 处理验证集
    input_json = os.path.join(folder_path,"data/dev.json")  # 原始文件路径
    output_json = os.path.join(folder_path,"data/dev_dealed.json")  # 输出文件路径
    remove_newlines_in_text(input_json, output_json)