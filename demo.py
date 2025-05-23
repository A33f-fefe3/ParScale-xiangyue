import json
from pydantic import KafkaDsn
import torch
import concurrent.futures
import re
from transformers import AutoModel, AutoTokenizer

class HierarchicalAggregationLayer(torch.nn.Module):
    def __init__(self, hidden_size=768, num_perspectives=3, attn_smoothing=0.1):
        """
        分层聚合层：通过三级处理融合多视角特征
        1) 两两视角组合的初级聚合
        2) 所有组合结果的全局聚合 
        3) 原始视角的注意力加权聚合
        
        参数：
            hidden_size (int): 特征向量的维度（默认768）
            num_perspectives (int): 需要聚合的视角数量（默认3）
            attn_smoothing (float): 注意力权重的标签平滑系数（默认0.1）
        """
        super().__init__()
        self.num_perspectives = num_perspectives
        self.attn_smoothing = attn_smoothing
        
        # 第一级聚合器：处理所有视角的两两组合
        # 共需要n*(n-1)/2个聚合器（n=视角数量）
        self.pairwise_aggregators = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),  # 将两个视角拼接后线性变换
                torch.nn.LayerNorm(hidden_size),               # 层归一化稳定训练
                torch.nn.GELU()                                # 高斯误差线性单元激活
            )
            for _ in range(num_perspectives * (num_perspectives - 1) // 2)
        ])
        
        # 第二级聚合器：整合所有两两组合的结果
        num_pairwise = num_perspectives * (num_perspectives - 1) // 2
        self.global_aggregator = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * num_pairwise, hidden_size * 2),  # 扩大维度增强表达能力
            torch.nn.LayerNorm(hidden_size * 2),                           # 归一化
            torch.nn.GELU(),                                               # 非线性激活
            torch.nn.Dropout(0.15),                                        # 随机失活防止过拟合
            torch.nn.Linear(hidden_size * 2, hidden_size)                  # 降维到原始维度
        )
        
        # 注意力机制：计算各原始视角的重要性权重
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * num_perspectives, hidden_size),  # 压缩视角拼接信息
            torch.nn.LayerNorm(hidden_size),                               # 归一化
            torch.nn.GELU(),                                              # 激活函数
            torch.nn.Linear(hidden_size, num_perspectives)                 # 输出各视角权重
        )
        
        self.softmax = torch.nn.Softmax(dim=-1)  # 将权重归一化为概率分布
        
    def forward(self, encoded_responses):
        """
        前向传播过程
        
        参数：
            encoded_responses: 多个视角的特征列表，每个元素形状为[hidden_size]
        
        返回：
            final_output: 聚合后的最终特征 [1, hidden_size]
            weights: 各视角的注意力权重 [1, num_perspectives] 
        """
        # 将多个视角特征堆叠为张量 [num_perspectives, hidden_size]
        stacked = torch.stack(encoded_responses)  
        
        # 第一阶段：两两视角聚合
        pairwise_outputs = []
        idx = 0  # 用于选择对应的聚合器
        
        # 遍历所有独特的视角组合对
        for i in range(self.num_perspectives):
            for j in range(i + 1, self.num_perspectives):
                # 拼接两个视角特征 [1, hidden_size*2]
                pair = torch.cat([stacked[i], stacked[j]], dim=-1).unsqueeze(0)
                
                # 通过对应的聚合器处理
                pairwise_outputs.append(self.pairwise_aggregators[idx](pair))
                idx += 1
        
        # 第二阶段：全局聚合
        # 拼接所有两两聚合结果 [1, hidden_size*num_pairs]
        pairwise_concat = torch.cat(pairwise_outputs, dim=-1)
        
        # 生成全局聚合特征 [1, hidden_size]
        global_repr = self.global_aggregator(pairwise_concat)
        
        # 第三阶段：注意力权重计算
        # 展平所有原始视角特征 [1, num_perspectives*hidden_size]
        concat = stacked.view(1, -1)
        
        # 计算归一化注意力权重 [1, num_perspectives]
        weights = self.softmax(self.attention(concat))
        
        # 应用标签平滑（正则化技术）
        if self.attn_smoothing > 0:
            uniform_dist = 1 / self.num_perspectives  # 均匀分布
            weights = weights * (1 - self.attn_smoothing) + self.attn_smoothing * uniform_dist
        
        # 计算原始视角的加权和 [hidden_size]
        weighted_aggregation = torch.sum(
            stacked * weights.t().unsqueeze(-1),  # 加权
            dim=0
        )
        
        # 最终输出：全局特征 + 加权原始特征 [1, hidden_size]
        final_output = global_repr + weighted_aggregation.unsqueeze(0)
        
        return final_output, weights

def load_encoder_model():
    """加载预训练的BERT编码器和分词器"""
    model_name = "/app/sda1/xiangyue/model/bert-base-chinese"
    return AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)

def encode_text(text, model, tokenizer):
    """
    将文本编码为向量表示
    使用BERT模型提取[CLS]标记作为文本的整体表示
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():  # 不计算梯度以提高推理效率
        # 提取[CLS]标记的嵌入作为文本表示
        return model(**inputs).last_hidden_state[:, 0, :].squeeze(0)

def process_perspective(index, question, encoder_model, tokenizer, llm, temperature=0.1):
    """
    处理单个视角：
    1. 调用大语言模型生成特定视角的回答
    2. 将回答编码为向量表示
    返回包含索引、回答文本和编码向量的元组
    
    参数:
        index: 视角索引
        question: 问题文本
        encoder_model: 编码器模型
        tokenizer: 分词器
        llm: 大语言模型调用函数
        temperature: 控制生成随机性的温度值
    """
    # 使用特定视角前缀提示大语言模型，并传入temperature参数
    response = llm(f"【换一种视角思考{index+1}】{question}", temperature=temperature)
    return index, response, encode_text(response, encoder_model, tokenizer)

def multi_perspective_analysis(metaprompt, p=3, topk=1, llm=None, temperature_settings=None):
    """
    多视角分析主函数：
    1. 并行生成多个视角的回答
    2. 对回答进行编码
    3. 使用增强聚合层计算各视角权重
    4. 选择权重最高的视角作为最终结果
    
    参数:
        metaprompt: 输入的问题/提示
        p: 视角数量
        topk: 返回topk个最佳视角
        llm: 大语言模型调用函数
        temperature_settings: 温度值设置，可以是单个值或长度为p的列表
    """
    # 如果未提供llm，则使用全局导入的llm_qwen
    if llm is None:
        llm = llm_qwen
    
    # 处理temperature设置
    if temperature_settings is None:
        temperatures = [0.1] * p  # 默认所有请求使用0.1
    elif isinstance(temperature_settings, (int, float)):
        temperatures = [temperature_settings] * p  # 单个值应用到所有请求
    elif isinstance(temperature_settings, (list, tuple)) and len(temperature_settings) == p:
        temperatures = temperature_settings  # 使用提供的温度列表
    else:
        raise ValueError("temperature_settings should be a single value or a list of length p")
    
    # 初始化聚合模型和编码器
    aggregation_model = HierarchicalAggregationLayer(hidden_size=768, num_perspectives=p)
    encoder_model, tokenizer = load_encoder_model()
    
    # 使用线程池并行处理多个，每个请求使用不同的temperature
    with concurrent.futures.ThreadPoolExecutor(max_workers=p) as executor:
        futures = [
            executor.submit(
                process_perspective, 
                i, 
                metaprompt, 
                encoder_model, 
                tokenizer, 
                llm, 
                temperatures[i]
            ) for i in range(p)
        ]
        results = [future.result() for future in futures]
    
    # 按索引排序确保顺序正确
    results.sort(key=lambda x: x[0])
    encoded_responses = [enc for _, _, enc in results]  # 提取编码向量
    response_texts = [resp for _, resp, _ in results]  # 提取回答文本
    for idx, (text, temp) in enumerate(zip(response_texts, temperatures), start=1):
        print(f"第{idx}个(temperature={temp:.2f})：{text}")
        
    if encoded_responses:
        # 聚合多视角信息并获取权重
        aggregated_output, weights = aggregation_model(encoded_responses)
        
        # 按权重排序视角
        sorted_items = sorted([(i, f"response_{i+1}", weights[0, i].item()) for i in range(p)], 
                             key=lambda x: x[2], reverse=True)
        
        # 获取topk视角
        top_indices = torch.topk(weights[0], topk).indices
        top_perspectives = [(f"response_{idx+1}", weights[0, idx].item(), response_texts[idx]) 
                           for idx in top_indices]
        
        # 构建最终标题（当前已注释掉）
        final_title = "\n".join(re.sub(r'【[^】]+】', '', content).strip() for _, _, content in top_perspectives)
        
        return {
            # "final_title": final_title,
            "top_perspectives": top_perspectives,
            # "temperature_settings": temperatures,  # 返回使用的温度设置
            # "all_weights": [(f"response_{i+1}", weights[0, i].item()) for i in range(p)]
        }
    return None

def llm_qwen(prompt, model="Qwen3", temperature=0.7, top_p=0.8, max_tokens=8192, presence_penalty=1.5):
    """
    修改后的LLM调用函数，增加temperature参数
    """
    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
    )
    return chat_response.choices[0].message.content

if __name__ == "__main__":
    from openai import OpenAI
    
    client = OpenAI(api_key="0", base_url="http://192.168.106.26:20000/v1")
    
    
    metaprompt = """
任务： 从给定的客服对话记录中提取关键信息，并生成一个结构化的 JSON 格式输出。

要求：
1. 提取以下字段：
  order_number（从系统消息或用户反馈中查找办理业务订单号）
  business_Number（用户要办理的手机号）
  name（用户提供的姓名，区分客服的姓名，和用户的姓名）
  contact_Phone（用户提供的手机号）
  user_feedback_issue（用户的主要投诉或咨询内容，尽量详细，且要以客服的视角描述）
  user_request（用户希望解决的问题或需求，尽量详细，且要以客服的视角描述）
  cons_progress（工号xx开头的客服为用户查询了什么，和用户确认了什么，尽量详细）
  cardType（判断用户咨询的业务类型号的卡，必须是这些门类：充值 权益包 流量包 业务办理 不确定）
2. 确保 JSON 格式正确，字段值尽量从对话原文中提取
3. 如果没有输出为空值！！！
最终输出格式：
{
  "order_number": "",
  "business_Number": "",
  "name": "",
  "contact_Phone": "",
  "user_feedback_issue": "",
  "user_request": "",
  "cons_progress": "",
  "cardType": ""
}
下面是要处理的客服对话片段：

(系统消息)      2025-05-11 14:13:04
正在为您接入人工服务，请稍等；您也可以发送【00】退出服务
(系统消息)      2025-05-11 14:13:04
客户IM号码（TYMY_H5dq250511024844208）已关联号码（TYMY_H515767962194）
梁鑫一(95047)      2025-05-11 14:13:05
您好，感谢您的耐心等待，系统已为您接入人工服务。中国电信95047号客服为您服务。
(0)      2025-05-11 14:13:05
【系统消息】15767962194,null,null ; 不可办理5G升级包
(0)      2025-05-11 14:13:05
人工[文本]
(0)      2025-05-11 14:13:05
【系统消息】亲，您可以#LinkIDSwitchToCSR马上召唤人工客服为您服务。<br/>以下是小知经常回答的问题哦：<br/>[1]购买号卡后找不到订单<br/>[2]我的卡有余额但无法使用<br/>[3]我想换个套餐有什么推荐吗<br/>[4]携号转网问题<br/>[5]我的卡提示被保护，停机了[文本]
(0)      2025-05-11 14:13:05
【客服请回复】请及时答复用户问题
(0)      2025-05-11 14:13:27
这个卡购买卡的30块不会作为预存话费的吗[文本]
梁鑫一(95047)      2025-05-11 14:13:49
尊敬的用户您好！这里是中国电信网上营业厅人工客服，工号95047，请问您有什么问题需要帮助吗？我将尽力为您提供最优质的服务！
梁鑫一(95047)      2025-05-11 14:13:53
辛苦您方便提供一下收货号码或身份证号吗？
(0)      2025-05-11 14:14:04
15767962194[文本]
(0)      2025-05-11 14:15:18
我昨天号码卡到了，快递小哥开卡后一直没信号，我打电话你们客服，客服让我再充值30元，但是现在只有30块话费，正常来说不应该是60块话费吗？[文本]
梁鑫一(95047)      2025-05-11 14:15:30
亲，正在为您积极查询中，预计需要1到3分钟，请您稍等。 （正在为您查询~!这句无需回复喔~）
梁鑫一(95047)      2025-05-11 14:17:02
15767962194未查询到信息，您是通过什么渠道购买的呢？方便提供订单编号吗？
梁鑫一(95047)      2025-05-11 14:17:16
抱歉亲亲查询错误
(0)      2025-05-11 14:17:17
你们这个小程序啊[文本]
梁鑫一(95047)      2025-05-11 14:17:19
2025-05-08 21:25:59下单的这个吗？
(0)      2025-05-11 14:17:23
对啊[文本]
梁鑫一(95047)      2025-05-11 14:17:47
亲，请您稍等不要离开呦~我现在为您查询，谢谢亲的配合与理解。
梁鑫一(95047)      2025-05-11 14:18:01
通用可用余额: 0专用余额: 30
梁鑫一(95047)      2025-05-11 14:18:02
您好，为了您能得到更专业的服务，您所咨询的问题需要转到号码归属（省）专席为您跟进（以上对话记录也会一同转接），转接期间请勿退出，如线路繁忙，可以联系号码归属省10000万号，同意转接回复1
(0)      2025-05-11 14:18:38
怎么回事？[文本]
(0)      2025-05-11 14:18:48
我就是问你怎么话费只有30块啊[文本]
(0)      2025-05-11 14:19:05
这个卡是买卡的30块不是作为预存话费的吗？[文本]
梁鑫一(95047)      2025-05-11 14:19:10
话费是您订单内的30激活已充值
(0)      2025-05-11 14:19:13
听不懂话吗？[文本]
(0)      2025-05-11 14:19:22
那我买卡的30块呢[文本]
梁鑫一(95047)      2025-05-11 14:19:33
这个就是买卡的亲亲
梁鑫一(95047)      2025-05-11 14:19:40
激活中要求您充值的这边查询不到
梁鑫一(95047)      2025-05-11 14:19:41
为您转接省内客服为您核实看看好吗？
(0)      2025-05-11 14:19:57
买卡的钱不会变成话费的意思是吗？[文本]
梁鑫一(95047)      2025-05-11 14:20:52
会
梁鑫一(95047)      2025-05-11 14:20:55
已经到了亲亲
(0)      2025-05-11 14:21:02
并没有啊[文本]
(0)      2025-05-11 14:21:15
我现在的30块是后面充值的啊[文本]
(0)      2025-05-11 14:21:24
现在你们两边踢皮球还是怎么样[文本]
梁鑫一(95047)      2025-05-11 14:21:36
亲，您说的问题这边为您记录反馈，请您提供一下能联系到您的手机号和您的姓名，给您带来不便请您谅解，我们会通过电话的方式告知您处理结果，请您保持电话畅通
梁鑫一(95047)      2025-05-11 14:21:54
您激活充值的30 账单麻烦截图
(0)      2025-05-11 14:22:22
SVU2OE1NIXhqeFMhUyFPTyFYO05IT1lqak5jeFlPbUNR1007-100-100-11-lnsy.png
梁鑫一(95047)      2025-05-11 14:22:47
亲亲稍等加载图片看下
(0)      2025-05-11 14:22:55
昨天充值的，赶紧看怎么回事啊，不然我就工信局投诉你们欺骗消费者了[文本]
梁鑫一(95047)      2025-05-11 14:24:45
15767962194这个号码能联系到您吗？
(0)      2025-05-11 14:25:29
可以[文本]
梁鑫一(95047)      2025-05-11 14:25:51
您好，您反馈的问题，客服已反馈到相关部门，预计48小时内，我们的工作人员会使用4008开头的电话跟您15767962194的号码联系，请您保持电话畅通，感谢您的理解！
梁鑫一(95047)      2025-05-11 14:27:44
亲亲您还有其他需要帮助的吗？
(系统消息)      2025-05-11 14:29:45
请问您还在线吗？ 如随后2分钟仍未收到您的消息，本次会话将暂时结束；如您的问题已解决，可发送【00】结束本次会话。
(系统消息)      2025-05-11 14:29:45
用户2分钟未发言或回复消息，系统已向用户发送提示消息
(系统消息)      2025-05-11 14:30:05
【系统消息】用户主动结束了该会话
"""
    
    # 手动设置每次并发的温度值
    manual_temperatures = [0.2, 0.4, 0.6, 0.8, 1, 1.4,1.8,2]
    # manual_temperatures = [0.2, 0.4, 0.6, 0.8, 1, 1.4, 2]
    # manual_temperatures = [0.2, 0.5, 0.8, 1, 1.4, 2]
    # manual_temperatures = [0.2, 0.6, 0.8, 1.4, 2]
    # manual_temperatures = [0.2, 0.6, 1, 2]
    # manual_temperatures = [0.2, 0.8, 2] # 客服总结
    # manual_temperatures = [0.2, 2] # topk = 1

    result = multi_perspective_analysis(
        metaprompt, 
        p=len(manual_temperatures),  # 自动根据温度列表长度确定p值
        topk=2, 
        llm=llm_qwen, 
        temperature_settings=manual_temperatures  # 传入手动设置的温度列表
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))  # 以JSON格式输出结果