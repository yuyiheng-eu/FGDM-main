# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

from signjoey.external_metrics import sacrebleu
from signjoey.external_metrics import mscoco_rouge
import numpy as np
from scipy.linalg import sqrtm
import torch
WER_COST_DEL = 3
WER_COST_INS = 3
WER_COST_SUB = 4


def chrf(references, hypotheses):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return (
        sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references).score * 100
    )


def bleu(references, hypotheses):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    bleu_scores = sacrebleu.raw_corpus_bleu(
        sys_stream=hypotheses, ref_streams=[references]
    ).scores
    scores = {}
    for n in range(len(bleu_scores)):
        scores["bleu" + str(n + 1)] = bleu_scores[n]
    return scores


def token_accuracy(references, hypotheses, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens) * 100 if all_tokens > 0 else 0.0


def sequence_accuracy(references, hypotheses):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum(
        [1 for (hyp, ref) in zip(hypotheses, references) if hyp == ref]
    )
    return (correct_sequences / len(hypotheses)) * 100 if hypotheses else 0.0


def rouge(references, hypotheses):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += mscoco_rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score * 100


def wer_list(references, hypotheses):
    total_error = total_del = total_ins = total_sub = total_ref_len = 0

    for r, h in zip(references, hypotheses):
        res = wer_single(r=r, h=h)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    return {
        "wer": wer,
        "del_rate": del_rate,
        "ins_rate": ins_rate,
        "sub_rate": sub_rate,
    }


def wer_single(r, h):
    r = r.strip().split()
    h = h.strip().split()
    edit_distance_matrix = edit_distance(r=r, h=h)
    alignment, alignment_out = get_alignment(r=r, h=h, d=edit_distance_matrix)

    num_cor = np.sum([s == "C" for s in alignment])
    num_del = np.sum([s == "D" for s in alignment])
    num_ins = np.sum([s == "I" for s in alignment])
    num_sub = np.sum([s == "S" for s in alignment])
    num_err = num_del + num_ins + num_sub
    num_ref = len(r)

    return {
        "alignment": alignment,
        "alignment_out": alignment_out,
        "num_cor": num_cor,
        "num_del": num_del,
        "num_ins": num_ins,
        "num_sub": num_sub,
        "num_err": num_err,
        "num_ref": num_ref,
    }


def edit_distance(r, h):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
    Main algorithm used is dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
    """
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
        (len(r) + 1, len(h) + 1)
    )
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                # d[0][j] = j
                d[0][j] = j * WER_COST_INS
            elif j == 0:
                d[i][0] = i * WER_COST_DEL
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + WER_COST_SUB
                insert = d[i][j - 1] + WER_COST_INS
                delete = d[i - 1][j] + WER_COST_DEL
                d[i][j] = min(substitute, insert, delete)
    return d


def get_alignment(r, h, d):
    """
    Original Code from https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    This function is to get the list of steps in the process of dynamic programming.
    Attributes:
        r -> the list of words produced by splitting reference sentence.
        h -> the list of words produced by splitting hypothesis sentence.
        d -> the matrix built when calculating the editing distance of h and r.
    """
    x = len(r)
    y = len(h)
    max_len = 3 * (x + y)

    alignlist = []
    align_ref = ""
    align_hyp = ""
    alignment = ""

    while True:
        if (x <= 0 and y <= 0) or (len(alignlist) > max_len):
            break
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] and r[x - 1] == h[y - 1]:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " * (len(r[x - 1]) + 1) + alignment
            alignlist.append("C")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif x >= 1 and y >= 1 and d[x][y] == d[x - 1][y - 1] + WER_COST_SUB:
            ml = max(len(h[y - 1]), len(r[x - 1]))
            align_hyp = " " + h[y - 1].ljust(ml) + align_hyp
            align_ref = " " + r[x - 1].ljust(ml) + align_ref
            alignment = " " + "S" + " " * (ml - 1) + alignment
            alignlist.append("S")
            x = max(x - 1, 0)
            y = max(y - 1, 0)
        elif y >= 1 and d[x][y] == d[x][y - 1] + WER_COST_INS:
            align_hyp = " " + h[y - 1] + align_hyp
            align_ref = " " + "*" * len(h[y - 1]) + align_ref
            alignment = " " + "I" + " " * (len(h[y - 1]) - 1) + alignment
            alignlist.append("I")
            x = max(x, 0)
            y = max(y - 1, 0)
        else:
            align_hyp = " " + "*" * len(r[x - 1]) + align_hyp
            align_ref = " " + r[x - 1] + align_ref
            alignment = " " + "D" + " " * (len(r[x - 1]) - 1) + alignment
            alignlist.append("D")
            x = max(x - 1, 0)
            y = max(y, 0)

    align_ref = align_ref[1:]
    align_hyp = align_hyp[1:]
    alignment = alignment[1:]

    return (
        alignlist[::-1],
        {"align_ref": align_ref, "align_hyp": align_hyp, "alignment": alignment},
    )

def fid_cpu(references, hypotheses):
    assert len(references) == len(hypotheses)
    fid_value = 0
    for pred, true in zip(hypotheses, references):
        # 均值和协方差
        mu1 = np.mean(true.cpu().numpy(), axis=0)
        sigma1 = np.cov(true, rowvar=False)
        mu2 = np.mean(pred.cpu().numpy(), axis=0)
        sigma2 = np.cov(pred, rowvar=False)

        # 计算均值差的平方
        mean_diff = np.sum((mu1 - mu2) ** 2)

        # 计算协方差的矩阵平方根
        covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)

        # 防止计算中产生复数
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 计算FID
        fid_value += mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)

    fid_value = fid_value / len(hypotheses)
    return fid_value

from scipy.linalg import sqrtm

def fid(references, hypotheses):
    assert len(references) == len(hypotheses)
    fid_value = 0.0
    
    if len(references) == 0:
        return 0.0

    device = references[0].device  # 获取数据所在的设备

    for pred, true in zip(hypotheses, references):
        # 0. 统一转换为 float64，保证和原 numpy 计算精度完全一致，避免浮点误差累积
        true_f64 = true.to(torch.float64)
        pred_f64 = pred.to(torch.float64)

        # 1. 均值计算 (GPU)
        mu1 = torch.mean(true_f64, dim=0)
        mu2 = torch.mean(pred_f64, dim=0)

        # 2. 协方差计算 (GPU)
        # 原代码 np.cov(..., rowvar=False) 是按列求协方差。
        # PyTorch 的 torch.cov 默认按行求协方差，因此这里必须转置 (.T) 才能保持效果完全不变
        sigma1 = torch.cov(true_f64.T)
        sigma2 = torch.cov(pred_f64.T)

        # 3. 均值差的平方 (GPU)
        mean_diff = torch.sum((mu1 - mu2) ** 2)

        # 4. 协方差矩阵的平方根 (先在GPU乘，再放CPU算sqrtm)
        # 这一步只拷贝 DxD 的小矩阵，耗时极低。为了保证底层的 Schur 分解数值完全一致，保留 scipy。
        sigma12 = torch.matmul(sigma1, sigma2)
        covmean_np, _ = sqrtm(sigma12.cpu().numpy(), disp=False)
        
        # 将算好的平方根矩阵转回 GPU 取实部
        covmean = torch.tensor(covmean_np.real, device=device, dtype=true_f64.dtype)

        # 5. 计算 FID
        batch_fid = (mean_diff + torch.trace(sigma1 + sigma2 - 2 * covmean)) / true.shape[0]
        
        # 使用 .item() 取出标量值，防止 Python 循环中计算图累积导致显存泄漏(OOM)
        fid_value += batch_fid.item()

    fid_value = fid_value / len(hypotheses)
    return fid_value

def mpjpe(references, hypotheses):
    # 两个列表的长度必须相等
    assert len(references) == len(hypotheses)
    mpjpe_value = 0
    # 迭代列表中的每一对对应张量
    for pred, true in zip(hypotheses, references):
        # 计算每个关节在每一帧的欧氏距离
        # distance = torch.norm(pred - true, dim=2)

        # 计算所有关节所有帧的平均距离
        # mpjpe = distance.mean()
        # 先沿一个轴检查，再沿另一个轴检查，这里先沿第1轴，然后再对结果沿第2轴检查

        first_all_zeros_index = np.where(np.all(true.numpy() == 0, axis=(1, 2)))[0]
        if first_all_zeros_index.size > 0:
            # 获取第一个全零帧的索引，由于我们要删除此帧及之后的所有帧，因此索引+1
            cut_index = first_all_zeros_index[0]
            true = true[:cut_index, :, :]
            pred = pred[:cut_index, :, :]
        #axis1_check = np.all(true.numpy() == 0, axis=1)
        #x = np.where(np.all(axis1_check, axis=1))[0]
        mpjpe_L = p_mpjpe(pred.numpy(), true.numpy())
        # 将 MPJPE 添加到列表中
        mpjpe_value += mpjpe_L

    return mpjpe_value


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True)) + 1e-8

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

def mpvpe(references, hypotheses):
    # 两个列表的长度必须相等
    assert len(references) == len(hypotheses)
    mpvpe_value = 0
    # 迭代列表中的每一对对应张量
    for pred, true in zip(hypotheses, references):

        # 计算每个顶点位置的欧氏距离
        vertex_errors = calculate_sequence_average_mpvpe(pred, true)

        # 计算所有顶点位置误差的平均值
        mpvpe_L = np.mean(vertex_errors)

        mpvpe_value += mpvpe_L

    return mpvpe_value


def calculate_sequence_average_mpvpe(predicted_sequence, ground_truth_sequence):
    """
    计算整个序列的平均Mean Per Vertex Position Error (MPVPE)。

    参数:
    - predicted_sequence: 预测的关节（或顶点）坐标序列，形状为(帧数, 关节数/顶点数, 3)。
    - ground_truth_sequence: 真实的关节（或顶点）坐标序列，形状同上。

    返回:
    - sequence_average_mpvpe: 整个序列的平均MPVPE得分。
    """
    # 确保输入形状一致
    assert predicted_sequence.shape == ground_truth_sequence.shape, "预测序列和真实序列形状不匹配"

    # 初始化总误差
    total_error = 0.0

    # 遍历序列中的每一帧
    for pred_frame, gt_frame in zip(predicted_sequence, ground_truth_sequence):
        # 计算单帧MPVPE并累加到总误差
        single_frame_mpvpe = calculate_single_frame_mpvpe(pred_frame, gt_frame)
        total_error += single_frame_mpvpe

    # 计算序列平均MPVPE
    sequence_average_mpvpe = total_error / predicted_sequence.shape[0]

    return sequence_average_mpvpe


def calculate_single_frame_mpvpe(predicted_frame, ground_truth_frame):
    """
    计算单帧内的Mean Per Vertex Position Error (MPVPE)。

    参数:
    - predicted_frame: 单帧的预测关节（或顶点）坐标，形状为(关节/顶点个数, 3)。
    - ground_truth_frame: 单帧的真实关节（或顶点）坐标，形状同上。

    返回:
    - single_frame_mpvpe: 单帧的MPVPE得分。
    """
    # 计算每个顶点位置的欧氏距离
    vertex_errors = np.linalg.norm(predicted_frame - ground_truth_frame, axis=1)
    # 计算平均误差
    single_frame_mpvpe = np.mean(vertex_errors)

    return single_frame_mpvpe

def calculate_angle(v1, v2):
    """
        计算两个向量之间的角度。
    """
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # 确保cos_theta在[-1, 1]之间，避免acos函数的域错误
    cos_theta = np.clip(cos_theta, -1., 1.)
    return np.arccos(cos_theta) * 180. / np.pi  # 将弧度转换为度


def mpjae(references, hypotheses):
    # 两个列表的长度必须相等
    assert len(references) == len(hypotheses)
    mpjae_value = 0
    # 迭代列表中的每一对对应张量
    for pred, true in zip(hypotheses, references):

        # 计算每个顶点位置的欧氏距离
        vertex_errors = mpjae_frame(pred, true, getSkeletalModelStructure())

        # 计算所有顶点位置误差的平均值
        mpjae_L = np.mean(vertex_errors)

        mpjae_value += mpjae_L

    return mpjae_value/len(hypotheses)


def mpjae_frame(predicted_joints, ground_truth_joints, joint_connections):
    """
    计算Mean Per Joint Angle Error。

    :param predicted_joints: 预测的关节坐标，形状为[帧数，关节数，3]
    :param ground_truth_joints: 真实的关节坐标，形状同上
    :param joint_connections: 关节连接关系列表，例如[(0, 1), (1, 2), ...]
    :return: MPJAE的平均值
    """
    frame_count, joint_count, _ = predicted_joints.shape
    assert predicted_joints.shape == ground_truth_joints.shape, "预测和真实关节坐标形状不匹配"

    total_errors = []
    first_all_zeros_index = np.where(np.all(ground_truth_joints.numpy() == 0, axis=(1, 2)))[0]
    if first_all_zeros_index.size > 0:
        frame_count = first_all_zeros_index[0]
    for frame in range(frame_count):
        frame_errors = []
        # 检查当前帧的所有关节坐标是否都接近0
        if np.allclose(predicted_joints[frame], 0) or np.allclose(ground_truth_joints[frame], 0):
            continue  # 跳过该帧
        for joint_pair in joint_connections:
            joint_i, joint_j = joint_pair
            pred_vec = predicted_joints[frame, joint_i] - predicted_joints[frame, joint_j]
            gt_vec = ground_truth_joints[frame, joint_i] - ground_truth_joints[frame, joint_j]

            pred_angle = calculate_angle(pred_vec, np.array([1, 0, 0]))  # 假设正X轴为基准
            gt_angle = calculate_angle(gt_vec, np.array([1, 0, 0]))

            # 计算角度差并取绝对值
            angle_error = abs(pred_angle - gt_angle)
            frame_errors.append(angle_error)

        # 对当前帧的所有关节角度误差求平均
        frame_errors_mean = np.mean(frame_errors)
        total_errors.append(frame_errors_mean)
        #filtered_list = [item for item in total_errors if not np.isnan(item)]
    # 对所有帧的误差求平均，得到最终的MPJAE
    mpjae_mean = np.mean(total_errors)
    return mpjae_mean

def getSkeletalModelStructure():
    return (
        # head
        (1, 0),

        (1, 2),

        # left arm
        (2, 3),

        (3, 4),      # 舍弃

        (1, 5),

        (5, 6),

        (6, 7),     # 舍弃

        (7, 8),

        (8, 9),

        (9, 10),

        (10, 11),

        (11, 12),

        (8, 13),

        (13, 14),

        (14, 15),

        (15, 16),

        (8, 17),

        (17, 18),

        (18, 19),

        (19, 20),

        (8, 21),

        (21, 22),

        (22, 23),

        (23, 24),

        (8, 25),

        (25, 26),

        (26, 27),

        (27, 28),

        (4, 29),

        (29, 30),

        (30, 31),

        (31, 32),

        (32, 33),

        (29, 34),

        (34, 35),

        (35, 36),

        (36, 37),

        (29, 38),

        (38, 39),

        (39, 40),

        (40, 41),

        (29, 42),

        (42, 43),

        (43, 44),

        (44, 45),

        (29, 46),

        (46, 47),

        (47, 48),

        (48, 49),

    )