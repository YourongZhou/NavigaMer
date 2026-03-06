import math
import random
from collections import defaultdict


class ALeSSpacedSeed:
    def __init__(
        self,
        weight,
        homology_length,
        similarity,
        k_seeds=1,
        upper_bound=None,
        random_seed=None,
        estimate_trials=100_000_000,
    ):
        # 初始化 ALeS 参数并预计算长度区间与灵敏度配置。
        if random_seed is not None:
            random.seed(random_seed)
        self.w = int(weight)
        self.N = int(homology_length)
        self.p = float(similarity)
        self.k = int(k_seeds)
        if self.w < 2:
            raise ValueError("weight 必须 >= 2")
        if self.k < 1:
            raise ValueError("k_seeds 必须 >= 1")
        if self.N <= self.w:
            raise ValueError("homology_length 必须大于 weight")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("similarity 必须在 [0,1]")
        self.upper_bound = int(upper_bound) if upper_bound is not None else self.N - 1
        if self.upper_bound < self.w:
            raise ValueError("upper_bound 必须 >= weight")
        if self.upper_bound >= self.N:
            self.upper_bound = self.N - 1
        self.m, self.M = self._precompute_min_max()
        self.original_m = self.m
        self.original_M = self.M
        self._sens_cache = {}
        self.estimate_trials = int(estimate_trials)
        if self.estimate_trials < 1:
            raise ValueError("estimate_trials 必须 >= 1")

    def _precompute_min_max(self):
        # 按论文回归式估计 seed 长度搜索区间 m/M。
        if self.p < 0.85:
            m = round(-0.8222 - 0.3194 * self.k + 1.1497 * self.w + 0.0699 * self.N)
            M = round(-0.4858 + 0.4339 * self.k + 1.2376 * self.w + 0.1586 * self.N)
        else:
            m = round(-0.8101 - 0.3352 * self.k + 1.1914 * self.w + 0.0581 * self.N)
            M = round(-1.1686 + 0.3576 * self.k + 1.4462 * self.w + 0.1366 * self.N)
        m = max(int(m), self.w)
        M = max(int(M), m)
        if m > self.upper_bound:
            m = self.upper_bound
        if M > self.upper_bound:
            M = self.upper_bound
        return m, M

    def _make_l(self, m, M):
        # 在给定区间内构造多 seed 的候选长度分布。
        if self.k == 1:
            return [random.randint(m, M)]
        cnt = [0] * 100
        lengths = [0] * self.k
        temp = M
        lengths[0] = m
        lengths[self.k - 1] = M
        cnt[lengths[0]] += 1
        cnt[lengths[self.k - 1]] += 1
        reached = False
        for i in range(1, self.k - 1):
            if not reached:
                lengths[i] = int(math.ceil((lengths[i - 1] + M) / 2.0))
            else:
                lengths[i] = temp
            bound = int(math.ceil(self.k / (pow(2.0, M - lengths[i] + 1))))
            if cnt[lengths[i]] < bound:
                cnt[lengths[i]] += 1
            elif cnt[lengths[i]] == bound:
                reached = True
                lengths[i] -= 1
                temp = lengths[i]
                cnt[lengths[i]] += 1
            else:
                reached = True
                cnt[lengths[i]] += 1
        lengths.sort()
        return lengths

    def _alloc_fixed_and_swap(self, lengths):
        # 生成固定骨架 seed 并通过 swap 降低重叠复杂度。
        seeds = []
        for length in lengths:
            chars = ["0"] * length
            chars[0] = "1"
            for j in range(length - self.w + 1, length):
                chars[j] = "1"
            seeds.append("".join(chars))
        return self._swap1_overlaps(seeds)

    def _random_seed_given_length(self, length):
        # 在指定长度下随机生成满足权重约束的单个 seed。
        chars = ["0"] * length
        chars[0] = "1"
        chars[length - 1] = "1"
        for j in range(2, self.w):
            pos = random.randint(1, length - j)
            idx = 0
            while pos > 0:
                if chars[idx] == "0":
                    pos -= 1
                idx += 1
            chars[idx - 1] = "1"
        return "".join(chars)

    def _random_seed_set(self, lengths):
        # 按长度列表批量生成随机 seed 集合。
        return [self._random_seed_given_length(length) for length in lengths]

    def _oc_value(self, seeds):
        # 计算 seed 集合在所有平移下的 overlap complexity。
        total = 0
        for i in range(len(seeds)):
            si = seeds[i]
            li = len(si)
            ones_i = [idx for idx, ch in enumerate(si) if ch == "1"]
            for j in range(i, len(seeds)):
                sj = seeds[j]
                lj = len(sj)
                for shift in range(-(lj - 1), li):
                    overlap = 0
                    for a in ones_i:
                        b = a - shift
                        if 0 <= b < lj and sj[b] == "1":
                            overlap += 1
                    total += 1 << overlap
        return total

    def _swap1_overlaps(self, seeds):
        # 执行 swap1 局部搜索以最小化 overlap complexity。
        seeds = list(seeds)
        weight = seeds[0].count("1")
        best_oc = self._oc_value(seeds)
        no_swaps = 0
        while True:
            found = False
            best_candidate = None
            for cur_seed in range(len(seeds)):
                chars = list(seeds[cur_seed])
                for i in range(1, len(chars) - 1):
                    if chars[i] != "1":
                        continue
                    for j in range(1, len(chars) - 1):
                        if chars[j] == "1":
                            continue
                        new_chars = chars[:]
                        new_chars[i] = "0"
                        new_chars[j] = "1"
                        new_seeds = seeds[:]
                        new_seeds[cur_seed] = "".join(new_chars)
                        oc = self._oc_value(new_seeds)
                        if oc < best_oc:
                            best_oc = oc
                            best_candidate = new_seeds
                            found = True
            if not found:
                break
            no_swaps += 1
            seeds = best_candidate
            if no_swaps >= weight * len(seeds):
                break
        return seeds

    @staticmethod
    def _bin_reversed_to_int(seed):
        # 将 seed 的反向二进制表示转换为整数编码。
        val = 0
        bit = 1
        for ch in seed:
            if ch == "1":
                val += bit
            bit <<= 1
        return val

    def _multiple_sensitivity_dp(self, seeds):
        # 使用自动机与动态规划精确计算多 seed 灵敏度。
        no_seeds = len(seeds)
        seed_lengths = [len(s) for s in seeds]
        max_l = max(seed_lengths)
        int_rev = [self._bin_reversed_to_int(s) for s in seeds]

        max_no_bs = no_seeds
        for seed in seeds:
            tmp = 1
            for ch in reversed(seed):
                if ch != "1":
                    tmp *= 2
                max_no_bs += tmp
        if max_no_bs > 50000:
            return self._estimate_sensitivity(seeds)

        nodes = []
        nodes.append({"b": 0, "left": -1, "right": 1, "suffix": 0, "hit": 0, "level": 0, "zero": 0})
        nodes.append({"b": 1, "left": -1, "right": -1, "suffix": 0, "hit": 0, "level": 1, "zero": 0})

        prev_start = 1
        prev_end = 1
        for level in range(2, max_l + 1):
            start_pos = len(nodes)
            for idx in range(prev_start, prev_end + 1):
                if nodes[idx]["hit"] == 1:
                    continue
                b = nodes[idx]["b"]
                for bit in (0, 1):
                    b_new = 2 * b + bit
                    compatible = False
                    hit = False
                    mask_bits = (1 << level) - 1
                    for seed_idx in range(no_seeds):
                        if seed_lengths[seed_idx] < level:
                            continue
                        prefix = int_rev[seed_idx] >> (seed_lengths[seed_idx] - level)
                        if (prefix & ((~b_new) & mask_bits)) == 0:
                            compatible = True
                            if level == seed_lengths[seed_idx]:
                                hit = True
                    if not compatible:
                        continue
                    new_node = {
                        "b": b_new,
                        "left": -1,
                        "right": -1,
                        "suffix": 0,
                        "hit": 1 if hit else 0,
                        "level": level,
                        "zero": 0,
                    }
                    if bit == 0:
                        nodes[idx]["left"] = len(nodes)
                        suffix = nodes[idx]["suffix"]
                        while suffix != 0 and nodes[suffix]["left"] == -1:
                            suffix = nodes[suffix]["suffix"]
                        if suffix != 0:
                            new_node["suffix"] = nodes[suffix]["left"]
                            if nodes[new_node["suffix"]]["hit"] == 1:
                                new_node["hit"] = 1
                    else:
                        nodes[idx]["right"] = len(nodes)
                        suffix = nodes[idx]["suffix"]
                        while nodes[suffix]["right"] == -1:
                            suffix = nodes[suffix]["suffix"]
                        new_node["suffix"] = nodes[suffix]["right"]
                        if nodes[new_node["suffix"]]["hit"] == 1:
                            new_node["hit"] = 1
                    nodes.append(new_node)
            prev_start = start_pos
            prev_end = len(nodes) - 1

        for i in range(1, len(nodes)):
            if nodes[i]["left"] != -1:
                nodes[i]["zero"] = nodes[i]["left"]
            else:
                zero = nodes[i]["suffix"]
                while zero != 0 and nodes[zero]["left"] == -1:
                    zero = nodes[zero]["suffix"]
                if zero != 0:
                    nodes[i]["zero"] = nodes[zero]["left"]

        f = [[0.0] * len(nodes) for _ in range(self.N + 1)]
        for i in range(self.N + 1):
            for j in range(len(nodes) - 1, -1, -1):
                node = nodes[j]
                if i == 0:
                    f[i][j] = 0.0
                elif i < node["level"]:
                    f[i][j] = 0.0
                elif node["hit"] == 1:
                    f[i][j] = 1.0
                else:
                    zero_link = node["zero"]
                    new_i = i - node["level"] + nodes[zero_link]["level"] - 1
                    if new_i < 0:
                        new_i = 0
                    f0 = f[new_i][zero_link]
                    if node["right"] < 0:
                        f1 = 1.0
                    else:
                        f1 = f[i][node["right"]]
                    f[i][j] = (1.0 - self.p) * f0 + self.p * f1
        return f[self.N][0]

    def _estimate_sensitivity(self, seeds, trials=None):
        # 通过蒙特卡洛采样估计 seed 集合灵敏度。
        if any(len(seed) >= self.N for seed in seeds):
            return 0.0
        if trials is None:
            trials = self.estimate_trials
        trials = int(trials)
        if trials < 1:
            raise ValueError("trials 必须 >= 1")

        seed_data = [(int(seed, 2), len(seed)) for seed in seeds]
        hits = 0
        for _ in range(trials):
            region = 0
            for _i in range(self.N):
                region = (region << 1) | (1 if random.random() < self.p else 0)
            found = False
            for seed_bits, seed_len in seed_data:
                for shift in range(0, self.N - seed_len + 1):
                    shifted = seed_bits << shift
                    if (region & shifted) == shifted:
                        found = True
                        break
                if found:
                    break
            if found:
                hits += 1
        return hits / float(trials)

    def _sensitivity(self, seeds):
        # 统一封装灵敏度计算并使用缓存避免重复计算。
        key = tuple(seeds)
        if key in self._sens_cache:
            return self._sens_cache[key]
        try:
            value = self._multiple_sensitivity_dp(seeds)
        except Exception:
            value = self._estimate_sensitivity(seeds)
        self._sens_cache[key] = value
        return value

    def _add_position(self, seeds, lengths):
        # 随机向某个 seed 插入一个 don't-care 位置并重优化。
        seeds = seeds[:]
        lengths = lengths[:]
        count = 0
        while True:
            count += 1
            if count == 20:
                return None, None
            seed_no = random.randrange(len(seeds))
            pos = random.randrange(lengths[seed_no])
            if pos != 0 and pos != lengths[seed_no] - 1 and lengths[seed_no] < self.N:
                break
        if lengths[seed_no] + 1 > self.upper_bound:
            return None, None
        seed = seeds[seed_no]
        seeds[seed_no] = seed[:pos] + "0" + seed[pos:]
        lengths[seed_no] += 1
        seeds = self._swap1_overlaps(seeds)
        lengths = [len(s) for s in seeds]
        return seeds, lengths

    def _remove_position(self, seeds, lengths):
        # 随机删除某个 seed 的 don't-care 位置并重优化。
        seeds = seeds[:]
        lengths = lengths[:]
        count = 0
        while True:
            count += 1
            if count == 20:
                return None, None
            seed_no = random.randrange(len(seeds))
            pos = random.randrange(lengths[seed_no])
            if seeds[seed_no][pos] == "0" and pos != 0 and pos != lengths[seed_no] - 1:
                break
        seed = seeds[seed_no]
        new_seed = seed[:pos] + seed[pos + 1 :]
        if len(new_seed) < self.w:
            return None, None
        seeds[seed_no] = new_seed
        lengths[seed_no] -= 1
        seeds = self._swap1_overlaps(seeds)
        lengths = [len(s) for s in seeds]
        return seeds, lengths

    def _find_optimal(self, seeds, lengths, cur_sens, trials=200):
        # 通过 add/remove 邻域搜索提升当前 seed 集合灵敏度。
        t_seeds = seeds[:]
        t_lengths = lengths[:]
        best_seeds = seeds[:]
        best_lengths = lengths[:]
        best_sens = cur_sens
        improved = False
        for i in range(1, trials + 1):
            if i == 1:
                continue
            choice = random.randrange(2)
            prev_seeds = t_seeds[:]
            prev_lengths = t_lengths[:]
            if choice == 1:
                new_seeds, new_lengths = self._add_position(t_seeds, t_lengths)
            else:
                new_seeds, new_lengths = self._remove_position(t_seeds, t_lengths)
            if new_seeds is None:
                t_seeds = prev_seeds
                t_lengths = prev_lengths
                continue
            val = self._sensitivity(new_seeds)
            if val > cur_sens:
                t_seeds = new_seeds
                t_lengths = new_lengths
                cur_sens = val
                best_sens = val
                best_seeds = new_seeds[:]
                best_lengths = new_lengths[:]
                improved = True
            else:
                t_seeds = prev_seeds
                t_lengths = prev_lengths
        if improved:
            return best_seeds, best_lengths, best_sens
        return seeds, lengths, cur_sens

    def _random_start_swap_for_oc_with_random_length(self, m, M, tries, best_sens, indel_trials=200):
        # 在随机长度起点上迭代执行 OC 优化与 indel 微调。
        old_m = m
        old_M = M
        bad_move = 0
        avg_m = 0.0
        avg_M = 0.0
        best_seeds = None
        best_lengths = None

        for k_iter in range(tries):
            bad_move += 1
            if k_iter % 50 == 0 and k_iter > 0:
                m = int(round(avg_m / 50.0))
                M = int(round(avg_M / 50.0))
                if bad_move == 49:
                    bad_move = 0
                    m = self.original_m
                    M = self.original_M
                avg_m = 0.0
                avg_M = 0.0
                old_m = m
                old_M = M
            else:
                m = old_m
                M = old_M
                if bad_move == 49:
                    bad_move = 0
                    m = self.original_m
                    M = self.original_M
                    old_m = m
                    old_M = M

            lengths = self._make_l(m, M)
            lengths = sorted(lengths)
            seeds = self._random_seed_set(lengths)
            seeds = self._swap1_overlaps(seeds)
            lengths = [len(s) for s in seeds]

            cur_sens = self._sensitivity(seeds)
            opt_seeds, opt_lengths, opt_sens = self._find_optimal(seeds, lengths, cur_sens, trials=indel_trials)
            if opt_sens > cur_sens:
                seeds = opt_seeds
                lengths = opt_lengths
                cur_sens = opt_sens

            min_len = min(len(s) for s in seeds)
            max_len = max(len(s) for s in seeds)
            avg_m += min_len
            avg_M += max_len

            if cur_sens > best_sens:
                bad_move = 0
                best_sens = cur_sens
                best_seeds = seeds[:]
                best_lengths = lengths[:]

        return best_sens, best_seeds, best_lengths

    def design_seeds(self, tries=500, indel_trials=200):
        # 执行 ALeS 自适应长度流程并返回当前最优 seed 集合。
        m, M = self.m, self.M
        if m < self.w:
            m = self.w
        if M < m:
            M = m

        best_sens = 0.0
        best_seeds = None
        best_lengths = None

        if self.k == 1 and self.upper_bound > self.w:
            m = self.w + 1

        if self.k > 1:
            init_lengths = self._make_l(m, M)
            init_seeds = self._alloc_fixed_and_swap(init_lengths)
            init_sens = self._sensitivity(init_seeds)
            best_sens = init_sens
            best_seeds = init_seeds[:]
            best_lengths = [len(s) for s in init_seeds]

        rs_sens, rs_seeds, rs_lengths = self._random_start_swap_for_oc_with_random_length(
            m, M, tries=tries, best_sens=best_sens, indel_trials=indel_trials
        )
        if rs_seeds is not None and rs_sens >= best_sens:
            best_sens = rs_sens
            best_seeds = rs_seeds
            best_lengths = rs_lengths
        if best_seeds is None:
            lengths = self._make_l(m, M)
            best_seeds = self._random_seed_set(lengths)
            best_sens = self._sensitivity(best_seeds)
            best_lengths = [len(s) for s in best_seeds]
        return {
            "seeds": best_seeds,
            "lengths": best_lengths,
            "sensitivity": best_sens,
            "m": self.m,
            "M": self.M,
        }

    @staticmethod
    def apply_mask(seq, mask):
        # 按 mask 提取序列中 care 位组成哈希键。
        return "".join(ch for ch, m in zip(seq, mask) if m == "1")

    def find_hits(self, query, reference, masks):
        # 用 spaced-seed 哈希索引在 query/reference 中查找命中。
        if isinstance(masks, str):
            masks = [masks]
        all_hits = []
        for mask in masks:
            mask_len = len(mask)
            if len(query) < mask_len or len(reference) < mask_len:
                continue
            ref_idx = defaultdict(list)
            for i in range(len(reference) - mask_len + 1):
                ref_key = self.apply_mask(reference[i : i + mask_len], mask)
                ref_idx[ref_key].append(i)
            for q in range(len(query) - mask_len + 1):
                q_key = self.apply_mask(query[q : q + mask_len], mask)
                if q_key in ref_idx:
                    for r in ref_idx[q_key]:
                        all_hits.append((mask, q, r))
        return all_hits

    def _ungapped_xdrop_extend(self, query, reference, q_seed, r_seed, seed_len, match_score=1, mismatch_score=-1, xdrop=5):
        # 从 seed 命中出发执行双向 ungapped X-drop 延伸。
        seed_score = 0
        for i in range(seed_len):
            seed_score += match_score if query[q_seed + i] == reference[r_seed + i] else mismatch_score

        best_left_gain = 0
        current_left = 0
        best_left_ext = 0
        step = 1
        while q_seed - step >= 0 and r_seed - step >= 0:
            current_left += match_score if query[q_seed - step] == reference[r_seed - step] else mismatch_score
            if current_left > best_left_gain:
                best_left_gain = current_left
                best_left_ext = step
            if best_left_gain - current_left > xdrop:
                break
            step += 1

        q_seed_end = q_seed + seed_len - 1
        r_seed_end = r_seed + seed_len - 1
        best_right_gain = 0
        current_right = 0
        best_right_ext = 0
        step = 1
        while q_seed_end + step < len(query) and r_seed_end + step < len(reference):
            current_right += match_score if query[q_seed_end + step] == reference[r_seed_end + step] else mismatch_score
            if current_right > best_right_gain:
                best_right_gain = current_right
                best_right_ext = step
            if best_right_gain - current_right > xdrop:
                break
            step += 1

        q_start = q_seed - best_left_ext
        r_start = r_seed - best_left_ext
        q_end = q_seed_end + best_right_ext
        r_end = r_seed_end + best_right_ext
        return {
            "q_start": q_start,
            "r_start": r_start,
            "q_end": q_end,
            "r_end": r_end,
            "score": seed_score + best_left_gain + best_right_gain,
            "aligned_length": q_end - q_start + 1,
        }

    def _local_hit_generation_1101(self, query, reference, anchor_hsp, window=64, min_triplets=3, min_len=8, match_score=1, mismatch_score=-1, xdrop=5):
        # 在锚点附近生成 1101 局部三重命中候选 HSP。
        q_lo = max(0, anchor_hsp["query_start"] - window)
        q_hi = max(q_lo, anchor_hsp["query_start"])
        r_lo = max(0, anchor_hsp["reference_start"] - window)
        r_hi = max(r_lo, anchor_hsp["reference_start"])

        diag_hits = defaultdict(list)
        # small model 1101: require match at offsets 0,1,3 (offset 2 is don't-care)
        for q in range(q_lo, max(q_lo, q_hi - 3)):
            for r in range(r_lo, max(r_lo, r_hi - 3)):
                if query[q] == reference[r] and query[q + 1] == reference[r + 1] and query[q + 3] == reference[r + 3]:
                    diag_hits[r - q].append((q, r))

        local_hsps = []
        for _, hits in diag_hits.items():
            hits.sort()
            if len(hits) < min_triplets:
                continue
            q0, r0 = hits[0]
            ext = self._ungapped_xdrop_extend(
                query,
                reference,
                q0,
                r0,
                4,
                match_score=match_score,
                mismatch_score=mismatch_score,
                xdrop=xdrop,
            )
            if ext["aligned_length"] >= min_len:
                local_hsps.append(
                    {
                        "query_start": ext["q_start"],
                        "query_end": ext["q_end"],
                        "reference_start": ext["r_start"],
                        "reference_end": ext["r_end"],
                        "score": ext["score"],
                        "mask": "1101(local)",
                    }
                )
        return local_hsps

    def _gap_link_cost(self, left_hsp, right_hsp, gap_open=-2, gap_extend=-1, mismatch_score=-1):
        # 计算允许重叠裁剪时两段 HSP 的拼接代价与裁剪信息。
        q_gap = right_hsp["query_start"] - left_hsp["query_end"] - 1
        r_gap = right_hsp["reference_start"] - left_hsp["reference_end"] - 1

        overlap_trim = max(0, -q_gap, -r_gap)
        left_len = min(
            left_hsp["query_end"] - left_hsp["query_start"] + 1,
            left_hsp["reference_end"] - left_hsp["reference_start"] + 1,
        )
        right_len = min(
            right_hsp["query_end"] - right_hsp["query_start"] + 1,
            right_hsp["reference_end"] - right_hsp["reference_start"] + 1,
        )

        max_trim_left = max(0, left_len - 1)
        max_trim_right = max(0, right_len - 1)
        if overlap_trim > max_trim_left + max_trim_right:
            return None

        trim_left = min(overlap_trim, max_trim_left)
        trim_right = overlap_trim - trim_left
        if trim_right > max_trim_right:
            trim_right = max_trim_right
            trim_left = overlap_trim - trim_right
            if trim_left > max_trim_left:
                return None

        q_gap_adj = q_gap + overlap_trim
        r_gap_adj = r_gap + overlap_trim
        if q_gap_adj < 0 or r_gap_adj < 0:
            return None

        indel_span = abs(q_gap_adj - r_gap_adj)
        gap_penalty = 0
        if indel_span > 0:
            gap_penalty += gap_open + gap_extend * indel_span

        adjust_penalty = mismatch_score * overlap_trim

        return {
            "cost": gap_penalty + adjust_penalty,
            "trim_left": trim_left,
            "trim_right": trim_right,
            "q_gap_adj": q_gap_adj,
            "r_gap_adj": r_gap_adj,
        }
    def _gapped_extension(self, query, reference, hsps, match_score=1, mismatch_score=-1, gap_open=-2, gap_extend=-1, local_window=64, retire_distance=128, diag_band=64, min_triplets=3):
        # 通过候选对角线链式拼接执行 PatternHunter 风格 gapped 扩展。
        if not hsps:
            return None

        nodes = sorted((dict(h) for h in hsps), key=lambda h: (h["query_start"], h["reference_start"], -h["score"]))
        for h in nodes:
            h["diag"] = h["reference_start"] - h["query_start"]
            h["chain_score"] = h["score"]
            h["prev"] = -1
            h["chain_q_start"] = h["query_start"]
            h["chain_r_start"] = h["reference_start"]
            h["trim_left"] = 0
            h["trim_right"] = 0

        base_count = len(nodes)
        local_index = {}

        def _register_local_node(local_hsp):
            # 将局部 1101 候选注册为可回溯的真实 HSP 节点。
            key = (
                local_hsp["query_start"],
                local_hsp["query_end"],
                local_hsp["reference_start"],
                local_hsp["reference_end"],
                local_hsp.get("mask", "1101(local)"),
            )
            idx = local_index.get(key)
            if idx is not None:
                return idx
            node = dict(local_hsp)
            node["diag"] = node["reference_start"] - node["query_start"]
            node["chain_score"] = node["score"]
            node["prev"] = -1
            node["chain_q_start"] = node["query_start"]
            node["chain_r_start"] = node["reference_start"]
            node["trim_left"] = 0
            node["trim_right"] = 0
            nodes.append(node)
            idx = len(nodes) - 1
            local_index[key] = idx
            return idx

        active = []
        retired = []

        for i in range(base_count):
            cur = nodes[i]

            still_active = []
            for j in active:
                left = nodes[j]
                if (
                    cur["query_start"] - left["query_end"] > retire_distance
                    and cur["reference_start"] - left["reference_end"] > retire_distance
                ):
                    retired.append((left["chain_score"], j))
                else:
                    still_active.append(j)
            active = still_active

            local_candidates = self._local_hit_generation_1101(
                query,
                reference,
                cur,
                window=local_window,
                min_triplets=min_triplets,
                match_score=match_score,
                mismatch_score=mismatch_score,
            )

            candidate_indices = [j for j in active if abs(nodes[j]["diag"] - cur["diag"]) <= diag_band]
            for lc in local_candidates:
                lc_idx = _register_local_node(lc)
                if abs(nodes[lc_idx]["diag"] - cur["diag"]) <= diag_band:
                    candidate_indices.append(lc_idx)

            for cand_idx in candidate_indices:
                cand = nodes[cand_idx]
                link_info = self._gap_link_cost(
                    cand,
                    cur,
                    gap_open=gap_open,
                    gap_extend=gap_extend,
                    mismatch_score=mismatch_score,
                )
                if link_info is None:
                    continue

                cand_chain_score = cand.get("chain_score", cand["score"])
                new_score = cand_chain_score + cur["score"] + link_info["cost"]
                if new_score > cur["chain_score"]:
                    cur["chain_score"] = new_score
                    cur["prev"] = cand_idx
                    cur["chain_q_start"] = cand.get("chain_q_start", cand["query_start"])
                    cur["chain_r_start"] = cand.get("chain_r_start", cand["reference_start"])
                    cur["trim_left"] = link_info["trim_left"]
                    cur["trim_right"] = link_info["trim_right"]

            active.append(i)

        retired.extend((nodes[j]["chain_score"], j) for j in active)
        if not retired:
            return None

        best_score, best_idx = max(retired, key=lambda x: x[0])

        chain = []
        seen = set()
        cur_idx = best_idx
        while cur_idx != -1 and cur_idx not in seen:
            seen.add(cur_idx)
            chain.append(cur_idx)
            cur_idx = nodes[cur_idx].get("prev", -1)
        chain.reverse()
        if not chain:
            return None

        start_trim = {idx: 0 for idx in chain}
        end_trim = {idx: 0 for idx in chain}
        for prev_idx, curr_idx in zip(chain[:-1], chain[1:]):
            edge = nodes[curr_idx]
            end_trim[prev_idx] = int(edge.get("trim_left", 0))
            start_trim[curr_idx] = int(edge.get("trim_right", 0))

        def _effective_bounds(idx):
            # 计算链上节点在首尾裁剪后的有效坐标范围。
            node = nodes[idx]
            qs = node["query_start"] + start_trim.get(idx, 0)
            rs = node["reference_start"] + start_trim.get(idx, 0)
            qe = node["query_end"] - end_trim.get(idx, 0)
            re = node["reference_end"] - end_trim.get(idx, 0)
            if qe < qs or re < rs:
                return None
            return qs, qe, rs, re

        bounds = []
        for idx in chain:
            b = _effective_bounds(idx)
            if b is None:
                continue
            bounds.append((idx, b))
        if not bounds:
            return None

        aligned_q = []
        aligned_r = []

        def _append_segment(qs, qe, rs, re):
            # 将一段 query/reference 片段按列追加到最终对齐字符串。
            q_len = max(0, qe - qs + 1)
            r_len = max(0, re - rs + 1)
            common = min(q_len, r_len)
            for offset in range(common):
                aligned_q.append(query[qs + offset])
                aligned_r.append(reference[rs + offset])
            for offset in range(common, q_len):
                aligned_q.append(query[qs + offset])
                aligned_r.append("-")
            for offset in range(common, r_len):
                aligned_q.append("-")
                aligned_r.append(reference[rs + offset])

        first_idx, (first_qs, first_qe, first_rs, first_re) = bounds[0]
        _append_segment(first_qs, first_qe, first_rs, first_re)

        prev_qe = first_qe
        prev_re = first_re
        for _idx, (cur_qs, cur_qe, cur_rs, cur_re) in bounds[1:]:
            q_gap = max(0, cur_qs - prev_qe - 1)
            r_gap = max(0, cur_rs - prev_re - 1)
            q_ptr = prev_qe + 1
            r_ptr = prev_re + 1

            common_gap = min(q_gap, r_gap)
            for offset in range(common_gap):
                aligned_q.append(query[q_ptr + offset])
                aligned_r.append(reference[r_ptr + offset])
            for offset in range(common_gap, q_gap):
                aligned_q.append(query[q_ptr + offset])
                aligned_r.append("-")
            for offset in range(common_gap, r_gap):
                aligned_q.append("-")
                aligned_r.append(reference[r_ptr + offset])

            _append_segment(cur_qs, cur_qe, cur_rs, cur_re)
            prev_qe = cur_qe
            prev_re = cur_re

        best_q_start = bounds[0][1][0]
        best_r_start = bounds[0][1][2]
        best_q_end = bounds[-1][1][1]
        best_r_end = bounds[-1][1][3]

        return {
            "q_start": best_q_start,
            "r_start": best_r_start,
            "q_end": best_q_end,
            "r_end": best_r_end,
            "score": best_score,
            "aligned_length": len(aligned_q),
            "mask": nodes[best_idx]["mask"],
            "aligned_query": "".join(aligned_q),
            "aligned_reference": "".join(aligned_r),
            "chain_indices": chain,
        }
    def align(self, query, reference, masks, match_score=1, mismatch_score=-1, xdrop=5, gap_open=-2, gap_extend=-1, local_window=64, retire_distance=128, diag_band=64, min_triplets=3):
        # 对单条 query 执行命中、延伸与拼接并返回最佳比对片段。
        hits = self.find_hits(query, reference, masks)
        if not hits:
            return None

        hsps = []
        for mask, q_pos, r_pos in hits:
            ext = self._ungapped_xdrop_extend(
                query,
                reference,
                q_pos,
                r_pos,
                len(mask),
                match_score=match_score,
                mismatch_score=mismatch_score,
                xdrop=xdrop,
            )
            hsps.append(
                {
                    "mask": mask,
                    "query_start": ext["q_start"],
                    "query_end": ext["q_end"],
                    "reference_start": ext["r_start"],
                    "reference_end": ext["r_end"],
                    "score": ext["score"],
                }
            )

        uniq = {}
        for h in hsps:
            k = (h["query_start"], h["query_end"], h["reference_start"], h["reference_end"], h["mask"])
            if k not in uniq or h["score"] > uniq[k]["score"]:
                uniq[k] = h
        hsps = list(uniq.values())

        gapped = self._gapped_extension(
            query,
            reference,
            hsps,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open=gap_open,
            gap_extend=gap_extend,
            local_window=local_window,
            retire_distance=retire_distance,
            diag_band=diag_band,
            min_triplets=min_triplets,
        )
        if gapped is None:
            return None

        q_start, q_end = gapped["q_start"], gapped["q_end"]
        r_start, r_end = gapped["r_start"], gapped["r_end"]
        return {
            "mask": gapped.get("mask", "N/A"),
            "query_start": q_start,
            "reference_start": r_start,
            "aligned_length": gapped["aligned_length"],
            "query_fragment": query[q_start : q_end + 1],
            "reference_fragment": reference[r_start : r_end + 1],
            "aligned_query": gapped.get("aligned_query", ""),
            "aligned_reference": gapped.get("aligned_reference", ""),
        }



if __name__ == "__main__":
    query = "ACGTAGCTAGCTAGCTG"
    reference = "TGCACGTAGCTAGCTAGCTGGAC"

    ales = ALeSSpacedSeed(
        weight=5,
        homology_length=20,
        similarity=0.8,
        k_seeds=1,
        # upper_bound=12,
        random_seed=7,
    )

    design = ales.design_seeds(tries=120, indel_trials=80)
    print(f"[ALeS配置] w={ales.w}, k={ales.k}, p={ales.p}, N={ales.N}, 区间=[{design['m']},{design['M']}]")
    print(">>> 设计出的最佳 seeds:")
    for seed in design["seeds"]:
        print(seed)
    print(f"Sensitivity={design['sensitivity']:.6f}")

    aln = ales.align(query, reference, design["seeds"])
    if aln is None:
        print("未找到命中。")
    else:
        print(
            f"最佳比对: mask={aln['mask']}, q_start={aln['query_start']}, "
            f"r_start={aln['reference_start']}, aligned_length={aln['aligned_length']}"
        )
