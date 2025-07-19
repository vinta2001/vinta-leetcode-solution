package com.vinta.leetcode;

import java.util.*;

public class Solution {

    /**
     * @param intervals
     * @return
     * @content 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
     * 请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
     * 示例 1：
     * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
     * 输出：[[1,6],[8,10],[15,18]]
     * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     * 示例 2：
     * 输入：intervals = [[1,4],[4,5]]
     * 输出：[[1,5]]
     * 解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
     * @url <a href="https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked">56. 合并区间</a>
     */
    public int[][] merge(int[][] intervals) {

    }

    /**
     * @param nums
     * @return
     * @content 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
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
