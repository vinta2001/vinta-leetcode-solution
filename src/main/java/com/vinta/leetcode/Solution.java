package com.vinta.leetcode;

import java.util.*;

public class Solution {

    /**
     * @param matrix
     * @describe 给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
     * 你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
     * @example1 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
     * 输出：[[7,4,1],[8,5,2],[9,6,3]]
     * @url <a href="https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked">48. 旋转图像</a>
     */
    public void rotate(int[][] matrix) {
        // j < i 且 matrix[i][j] <=> matrix[j][i] => 左下角和右上角交换元素
        // j < n - 1 - i 且 matrix[i][j] <=> matrix[j][i] => 左上角和右下角交换元素
        // j < n / 2 且 matrix[i][j] <=> matrix[i][n-1-j] => 左右交换元素
        // i < n / 2 且 matrix[i][j] <=> matrix[n-1-i][j] => 上下交换元素
        int n = matrix.length;

        // 交换左下与右上
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        // 交换左右
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = temp;
            }
        }

    }

    /**
     * @param matrix
     * @return
     * @describe 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
     * @example1 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
     * 输出：[1,2,3,6,9,8,7,4,5]
     * @url <a href="https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked">54. 螺旋矩阵</a>
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;

        while (true) {
            for (int i = left; i <= right; i++) res.add(matrix[top][i]);  //从左到右
            if (++top > bottom) break;
            for (int i = top; i <= bottom; i++) res.add(matrix[i][right]); //从上到下
            if (--right < left) break;
            for (int i = right; i >= left; i--) res.add(matrix[bottom][i]); //从右到左
            if (--bottom < top) break;
            for (int i = bottom; i >= top; i--) res.add(matrix[i][left]); //从下到上
            if (++left > right) break;
        }
        return res;
    }

    /**
     * @param matrix
     * @describe 给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
     * @example1 输入：matrix =
     * [[1,1,1],
     * [1,0,1],
     * [1,1,1]]
     * 输出：
     * [[1,0,1],
     * [0,0,0],
     * [1,0,1]]
     * @url <a href="https://leetcode.cn/problems/set-matrix-zeroes/?envType=study-plan-v2&envId=top-100-liked">73. 矩阵置零</a>
     */
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        boolean[] rows = new boolean[m];
        boolean[] cols = new boolean[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i][j] == 0) {
                    rows[i] = true;
                    cols[j] = true;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (cols[j] || rows[i]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    /**
     * @param nums
     * @return
     * @describe 给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。
     * 请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
     * @example1： 输入：nums = [1,2,0]
     * 输出：3
     * 解释：范围 [1,2] 中的数字都在数组中。
     * @example2： 输入：nums = [3,4,-1,1]
     * 输出：2
     * 解释：1 在数组中，但 2 没有。
     * @example3： 输入：nums = [7,8,9,11,12]
     * 输出：1
     * 解释：最小的正数 1 没有出现。
     * @url <a href="https://leetcode.cn/problems/first-missing-positive/description/?envType=study-plan-v2&envId=top-100-liked">41. 缺失的第一个正数</a>
     */
    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        int[] existing = new int[len + 1]; //保存 [0,len] 区间的正数
        for (int num : nums) {
            if (num >= 0 && num <= len) {
                existing[num]++;
            }
        }
        for (int i = 1; i < existing.length; i++) {
            if (existing[i] == 0) {
                return i;
            }
        }
        return existing.length; //如果[0,len]区间的数都在，那么就是len+1不存在
    }

    /**
     * @param nums
     * @return
     * @describe 给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。
     * 题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。
     * 请 不要使用除法，且在 O(n) 时间复杂度内完成此题。
     * @example1: 输入: nums = [1,2,3,4]
     * 输出: [24,12,8,6]
     * @example2: 输入: nums = [-1,1,0,-3,3]
     * 输出: [0,0,9,0,0]
     * @url <a href="https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2&envId=top-100-liked">238. 除自身以外数组的乘积</a>
     */

    public int[] productExceptSelf(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        Arrays.fill(res, 1);

        for (int i = 1; i < len; i++) {
            res[i] = res[i - 1] * nums[i - 1];  // res[i]表示除nums[i]以外的乘积
        }

        int suf = 1;
        for (int i = len - 1; i >= 0; i--) {
            res[i] *= suf;   // 先乘suf，表示在乘nums[i]之前就完成乘积，所以没有乘nums[i]
            suf *= nums[i];
        }

        return res;
    }

    /**
     * @param nums
     * @param k
     * @describe 给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。
     * @example1: 输入: nums = [1,2,3,4,5,6,7], k = 3
     * 输出: [5,6,7,1,2,3,4]
     * 解释:
     * 向右轮转 1 步: [7,1,2,3,4,5,6]
     * 向右轮转 2 步: [6,7,1,2,3,4,5]
     * 向右轮转 3 步: [5,6,7,1,2,3,4]
     * @example2: 输入：nums = [-1,-100,3,99], k = 2
     * 输出：[3,99,-1,-100]
     * 解释:
     * 向右轮转 1 步: [99,-1,-100,3]
     * 向右轮转 2 步: [3,99,-1,-100]
     * @url <a href="https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked">189. 轮转数组</a>
     */
    public void rotate(int[] nums, int k) {
        int len = nums.length;
        k %= len; // k % len
        if (len < k) return;
        rotateHelp(nums, 0, len - 1);
        rotateHelp(nums, 0, k - 1);
        rotateHelp(nums, k, len - 1);
    }

    public void rotateHelp(int[] nums, int s, int e) {
        while (s < e) {
            int temp = nums[s];
            nums[s] = nums[e];
            nums[e] = temp;
            s++;
            e--;
        }
    }

    /**
     * @param intervals
     * @return
     * @describe 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
     * @example1 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
     * 输出：[[1,6],[8,10],[15,18]]
     * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     * @example2 输入：intervals = [[1,4],[4,5]]
     * 输出：[[1,5]]
     * 解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
     * @url <a href="https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked">56. 合并区间</a>
     */
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        LinkedList<int[]> res = new LinkedList<>();
        res.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] <= res.getLast()[1]) {
                res.getLast()[1] = Math.max(intervals[i][1], res.getLast()[1]);
            } else {
                res.add(intervals[i]);
            }
        }
        return res.toArray(new int[0][]);
    }

    /**
     * @param nums
     * @return
     * @describe 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
     * <p>
     * 子数组是数组中的一个连续部分。
     * @excample1 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
     * 输出：6
     * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
     * @url <a href="https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked">53. 最大子数组和</a>
     */
    public int maxSubArray(int[] nums) {
        if (nums.length == 1) return nums[0];
        if (nums.length == 0) return 0;
        int sum = Integer.MIN_VALUE / 2;
        int maxSum = Integer.MIN_VALUE / 2;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            sum = Math.max(sum, nums[i]);
            maxSum = Math.max(maxSum, sum);
        }
        return maxSum;
    }

    /**
     * @param s
     * @param t
     * @return
     * @describe 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
     * 注意：
     * 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
     * 如果 s 中存在这样的子串，我们保证它是唯一的答案。
     * @example1 输入：s = "ADOBECODEBANC", t = "ABC"
     * 输出："BANC"
     * 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
     * @example2 输入：s = "a", t = "a"
     * 输出："a"
     * 解释：整个字符串 s 是最小覆盖子串。
     * @url <a href="https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2&envId=top-100-liked">76.最小覆盖子串</a>
     */
    public String minWindow(String s, String t) {
        int[] cntT = new int[128];
        int[] cntS = new int[128];
        int lessCnt = 0; // 字符个数，不计次数，在后面的过程中处理
        for (char c : t.toCharArray()) {
            if (cntT[c]++ == 0) { // 先比较，再递增
                lessCnt++;
            }
        }
        int len = s.length();
        int ansL = -1, ansR = len;
        int left = 0;
        char[] chars = s.toCharArray();
        for (int right = 0; right < len; right++) {
            char c = chars[right];
            if (++cntS[c] == cntT[c]) {  // 如果出现次数相同，则去掉该字母，表示被覆盖
                lessCnt--;
            }
            while (lessCnt == 0) {
                if ((right - left) < (ansR - ansL)) {
                    ansR = right;
                    ansL = left;
                }
                char x = chars[left++];
                if (cntS[x]-- == cntT[x]) {  // 数量减到相同，则表示还有字符没有被覆盖
                    lessCnt++;
                }
            }
        }
        return ansL < 0 ? "" : s.substring(ansL, ansR + 1);
    }

    /**
     * @param nums
     * @param k
     * @return
     * @leetcode 239
     * @title 滑动窗口最大值
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - (k - 1)]; //从第k个开始有窗口，第k个下标是k-1
        //维护一个单调递减队，一边弹出，一边压入，单调队列里面可以存下标也可以存数组中的值

        /**
         * 单调队列的原则：如果新值与队列不单调，则弹出小的值；如果新值与队列单调，则直接加入
         */

        // 这里存入的是下标，便于记录窗口
        Deque<Integer> queue = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            while (!queue.isEmpty() && nums[queue.getLast()] <= nums[i]) {
                queue.removeLast();
            }
            queue.addLast(i);

            int left = i - k + 1;
            if (queue.getFirst() < left) {
                queue.removeFirst();
            }
            if (left >= 0) {
                res[left] = nums[queue.getFirst()];
            }
        }
        return res;
    }

    /**
     * @param nums
     * @param k
     * @return
     * @url <a href="https://leetcode.cn/problems/subarray-sum-equals-k/?envType=study-plan-v2&envId=top-100-liked">560.和为 K 的子数组</a>
     */
    public int subarraySum(int[] nums, int k) {
        // 前i个数字的前缀和 prefix[i]（当前前缀和）
        // prefix[i] - prefix[j] = k; (j < i, prefix[j] < prefix[i])
        // prefix[j]=prefix[i]-k;

        // map记录前缀和的个数
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        int res = 0;
        int prefix = 0;
        for (int num : nums) {
            prefix += num;
            res += map.getOrDefault(prefix - k, 0);
            map.merge(prefix, 1, Integer::sum);  //map[prefix]++;
        }
        return res;
    }


    /**
     * @param s
     * @param p
     * @return
     * @title 找到字符串中所有字母异位词
     * @leetcode 438
     */
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int[] ch1 = new int[26];
        int[] ch2 = new int[26];
        for (char c : p.toCharArray()) {
            ch1[c - 'a']++;
        }
        char[] chars = s.toCharArray();
        for (int i = 0; i < s.length(); i++) {
            ch2[chars[i] - 'a']++;
            int j = i + 1 - p.length();
            if (j < 0) continue;
            if (Arrays.equals(ch1, ch2)) {
                res.add(j);
            }
            ch2[chars[j] - 'a']--;
        }
        return res;
    }

    /**
     * @param s
     * @return
     * @titile 无重复字符的最长子串
     * @leetcode 3
     */
    public int lengthOfLongestSubstring(String s) {
        int res = 0;
        int l = 0;
        Map<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();
        for (int r = 0; r < chars.length; r++) {
            if (map.containsKey(chars[r])) {
                // 取最大的位置，防止重复的字符在考前的位置
                // s = "abba"时的第2个"a"
                l = Math.max(map.get(chars[r]) + 1, l);
            }
            map.put(chars[r], r);
            res = Math.max(res, r - l + 1);
        }
        return res;
    }

    /**
     * @param height
     * @return
     * @title 接雨水
     * @leetcode 42
     */
    public int trap(int[] height) {
        int left = 0, leftMax = 0;
        int right = height.length - 1, rightMax = 0;
        int res = 0;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (height[left] < height[right]) {
                // 如果 leftMax = height[left]
                // 那么 leftMax < height[right]
                // 所以左边的槽一定可以接住水
                res += leftMax - height[left];
                left++;
            } else {
                // 同理如果 rightMax = height[right]
                // 那么 rightMax  <= height[left]
                // 所以右边的槽一定可以接住水
                res += rightMax - height[right];
                right--;
            }
        }
        return res;
    }

    /**
     * @param nums
     * @return
     * @title 三数之和
     * @leetcode 15
     */
    public List<List<Integer>> threeSum(int[] nums) {
        int len = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            if (i > 0 && nums[i - 1] == nums[i]) continue;
            int j = i + 1, k = len - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    while (k > j && nums[--k] == nums[k + 1]) ;
                    while (j < k && nums[++j] == nums[j - 1]) ;
                } else if (sum > 0) {
                    while (j < k && nums[k] == nums[--k]) ;
                } else {
                    while (j < k && nums[j] == nums[++j]) ;
                }
            }
        }
        return res;
    }

    /**
     * 盛水最多的容器 leetcode-11
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        int lid = 0, rid = height.length - 1;
        int res = 0;
        while (lid < rid) {
            if (height[lid] < height[rid]) {
                res = Math.max(res, height[lid] * (rid - lid));
                lid++;
            } else {
                res = Math.max(res, height[rid] * (rid - lid));
                rid--;
            }
        }
        return res;
    }

    /**
     * @param nums
     * @title 移动零
     * @leetcode 283
     */
    public void moveZeroes(int[] nums) {
        int ptr = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[ptr];
                nums[ptr] = temp;
                ptr++;
            }
        }

    }

    /**
     * @param nums
     * @return
     * @title 最长连续序列
     * @leetcode 128
     */
    public int longestConsecutive(int[] nums) {
        int ans = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int num : set) {
            if (set.contains(num - 1)) {  // 已经被计算过是连续子序列的一部分
                continue;
            }
            int y = num + 1;
            while (set.contains(y)) {
                y++;
            }
            ans = Math.max(ans, y - num);
        }
        return ans;
    }

    /**
     * @param strs
     * @return
     * @title 字母异位词分组
     * @leetcode 49
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> record = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = new String(chars);
            List<String> values = record.getOrDefault(key, new ArrayList());
            values.add(str);
            record.put(key, values);
        }
        return new ArrayList<>(record.values());
    }

    /**
     * @param nums
     * @param target
     * @return
     * @title 两数之和
     * @leetcode 1
     */
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        int[] res = new int[2];
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            } else {
                map.put(nums[i], i);
            }
        }
        return res;
    }

    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        // char[] map = new char[s.length()];
        Map<Character, Character> s2t = new HashMap<>();
        Map<Character, Character> t2s = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if ((s2t.containsKey(s.charAt(i)) && (s2t.get(s.charAt(i)) != t.charAt(i))) ||
                    t2s.containsKey(t.charAt(i)) && (t2s.get(t.charAt(i)) != s.charAt(i))) {
                return false;
            } else {
                t2s.put(t.charAt(i), s.charAt(i));
                s2t.put(s.charAt(i), t.charAt(i));
            }
        }
        return true;
    }

    public List<Integer> findSubstring(String s, String[] words) {
        int wordLen = words[0].length(); // 一个单词的长度
        int windowLen = wordLen * words.length; // 所有单词的总长度，即窗口大小

        // 目标：窗口中的单词出现次数必须与 targetCnt 完全一致
        Map<String, Integer> targetCnt = new HashMap<>();
        for (String w : words) {
            targetCnt.merge(w, 1, Integer::sum); // targetCnt[w]++
        }

        List<Integer> ans = new ArrayList<>();
        // 枚举窗口起点，做 wordLen 次滑动窗口
        for (int start = 0; start < wordLen; start++) {
            Map<String, Integer> cnt = new HashMap<>();
            int overload = 0;
            // 枚举窗口最后一个单词的右端点+1
            for (int right = start + wordLen; right <= s.length(); right += wordLen) {
                // 1. inWord 进入窗口
                String inWord = s.substring(right - wordLen, right);
                // 下面 cnt[inWord]++ 后，inWord 的出现次数过多
                if (cnt.getOrDefault(inWord, 0).equals(targetCnt.getOrDefault(inWord, 0))) {
                    overload++;
                }
                cnt.merge(inWord, 1, Integer::sum); // cnt[inWord]++

                int left = right - windowLen; // 窗口第一个单词的左端点
                if (left < 0) { // 窗口大小不足 windowLen
                    continue;
                }

                // 2. 更新答案
                // 如果没有超出 targetCnt 的单词，那么也不会有少于 targetCnt 的单词
                if (overload == 0) {
                    ans.add(left);
                }

                // 3. 窗口最左边的单词 outWord 离开窗口，为下一轮循环做准备
                String outWord = s.substring(left, left + wordLen);
                cnt.merge(outWord, -1, Integer::sum); // cnt[outWord]--
                if (cnt.get(outWord).equals(targetCnt.getOrDefault(outWord, 0))) {
                    overload--;
                }
            }
        }

        return ans;
    }

    public int minPathSum(int[][] matrix) {
        // write code here
        int m = matrix.length;
        int n = matrix[0].length;

        int[][] dp = new int[m][n];
        dp[0][0] = matrix[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] += dp[i - 1][0] + matrix[i][0];
        }
        for (int j = 1; j < n; j++) {
            dp[0][j] += dp[0][j - 1] + matrix[0][j];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = matrix[i][j] + Math.min(dp[i][j - 1], dp[i - 1][j]);
            }
        }
        return dp[m - 1][n - 1];
    }

    public int search(int[] nums, int target) {
        // write code here
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = (l + r) >> 1;
            if (nums[m] == target) {
                return m;
            }
            if (nums[m] >= nums[l]) {   // k在右侧: l,m,k,r
                if (target < nums[m] && target >= nums[l]) { // target在[l, m)之间
                    r = m;
                } else { // target在[m, r]之间
                    l = m + 1;
                }
            } else if (nums[m] <= nums[r]) { // k在m左侧: l,k,m,r
                if (target >= nums[m] && target <= nums[r]) {  // target在(m, r]之间
                    l = m + 1;
                } else {
                    r = m;
                }
            }
        }

        return -1;
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows == 0) return res;
        res.add(Arrays.asList(1));
        if (numRows == 1) return res;
        res.add(Arrays.asList(1, 1));
        if (numRows == 2) return res;
        for (int i = 1; i < numRows - 1; i++) {
            List<Integer> cur = res.get(i);
            List<Integer> tmp = new ArrayList<>();
            tmp.add(1);
            for (int j = 0; j < cur.size() - 1; j++) {
                tmp.add(cur.get(j) + cur.get(j + 1));
            }
            tmp.add(1);
            res.add(tmp);
        }

        return res;
    }

    public int coinChange(int[] coins, int amount) {
        int res = 0;
        int idx = coins.length - 1;
        Arrays.sort(coins);
        while (amount > 0 && idx >= 0) {
            if (amount >= coins[idx]) {
                amount -= coins[idx];
                res++;
            } else {
                idx--;
            }
        }
        return amount > 0 ? -1 : res;
    }

    public void zeroOneBag(int[] costs, int[] val, int target) {
        int[] dp = new int[target + 1];
        for (int i = 0; i < costs.length; i++) {
            for (int j = target; j > costs[i]; j++) {
                dp[j] = Math.max(dp[j - costs[i]] + val[i], dp[j]);
            }
        }

        Map<Integer, Integer> map = new HashMap<>();
    }

    public int findDuplicate(int[] nums) {
        int slow = 0;
        int fast = 0;
        while (true) {
            fast = nums[fast];
            fast = nums[fast];

            slow = nums[slow];
            if (slow == fast) break;
        }
        int ptr = 0;
        while (ptr != slow) {
            ptr = nums[ptr];
            slow = nums[slow];
        }
        return ptr;
    }
}
