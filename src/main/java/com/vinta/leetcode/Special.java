package com.vinta.leetcode;

import java.util.*;
import java.util.stream.Collectors;

public class Special {
    public static void main(String[] args) {
//        System.out.println(maxN(new int[]{1, 2, 4, 9}, 2533));
        System.out.println(stringNumberAdd("1", "99"));
        System.out.println(mostCountLetter("abbba"));
        System.out.println(intersection(new int[]{1, 3, 5, 7, 9, 11}, new int[]{2, 3, 5, 7, 11}));
        System.out.println(KthListNode(SolutionUtils.createListNode(new Integer[]{1, 2}), 2));
    }

    /**
     * 小于N的最大数
     *
     * @param digits
     * @param n
     * @return
     */
    public static int maxN(int[] digits, int n) {
        int res = 0;
        Arrays.sort(digits);
        String s = String.valueOf(n);
        boolean less = false;
        for (int i = 0; i < s.length(); i++) {
            if (less) {
                res = res * 10 + digits[digits.length - 1];
                continue;
            }
            int target = s.charAt(i) - '0';
            int num = binarySearch(digits, target, i < s.length() - 1 ? s.charAt(i + 1) - '0' : digits[0]);
            if (num < target) {
                res = res * 10 + num;
                less = true;
            } else if (num == target) {
                res = res * 10 + num;
            } else return -1;
        }
        return res;
    }

    private static int binarySearch(int[] nums, int target, int next) {
        if (next < nums[0]) target--;
        int l = 0, r = nums.length - 1;
        while (l <= r) {
            int m = (l + r) >> 1;
            if (r - l <= 1) {
                if (nums[r] <= target) return nums[r];
                return nums[l];
            } else if (nums[m] == target) {
                return nums[m];
            } else if (nums[m] > target) {
                r = m - 1;
            } else {
                l = m;
            }
        }
        return nums[l];
    }

    /**
     * 无重复字符的最长子串
     *
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int left = 0;
        int res = 0;
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; i++) {
            if (map.containsKey(chars[i])) {
                // 取最大的位置，防止重复的字符靠前的位置
                // 如s = "abba" 时的第2个"a"时，left = 3 而不是 left = 0
                left = Math.max(left, map.get(chars[i]) + 1);
            }
            map.put(chars[i], i);
            res = Math.max(res, i - left + 1);
        }
        return res;
    }

    /**
     * 统计字符串中出现次数最多的字母及其对应的次数
     *
     * @param s
     * @return
     */
    public static String mostCountLetter(String s) {
        Map<Character, Integer> map = new HashMap<>();
        char[] chars = s.toCharArray();
        for (char c : chars) {
            map.merge(c, 1, Integer::sum);
        }
        char resChar = '0';
        int resCount = 0;
        for (Map.Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getValue() > resCount) {
                resChar = entry.getKey();
                resCount = entry.getValue();
            }
        }
        return "{ " + resChar + " : " + resCount + " }";
    }

    /**
     * 大数相加
     *
     * @param s
     * @param t
     * @return
     */
    public static String stringNumberAdd(String s, String t) {
        if (s.length() < t.length()) {
            String tmp = t;
            t = s;
            s = tmp;
        }
        int sum = 0;
        char[] res = new char[s.length()];
        for (int i = s.length() - 1; i >= 0; i--) {
            sum += s.charAt(i) - '0';
            int j = i - (s.length() - t.length());
            if (j >= 0) {
                sum += t.charAt(j) - '0';
            }
            res[i] = (char) (sum % 10 + '0');
            sum /= 10;
        }
        String output = String.valueOf(res);
        if (sum >= 0) {
            output = sum + output;
        }
        return output;
    }

    /**
     * 集合交集
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public static List intersection(int[] nums1, int[] nums2) {
        Set<Integer> set1 = Arrays.stream(nums1).boxed().collect(Collectors.toSet());
        Set<Integer> set2 = Arrays.stream(nums2).boxed().collect(Collectors.toSet());
        List<Integer> res = new ArrayList<>();
        for (int num : set1) {
            if (set2.contains(num)) {
                res.add(num);
            }
        }
        return res;
    }

    /**
     * 倒数第k个数
     *
     * @param head
     * @return
     */
    public static int KthListNode(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;

        while (k-- >= 0) {
            fast = fast.next;
        }

        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow.next.val;
    }

    /**
     * 岛屿数量
     *
     * @param grid
     * @return
     */
    public static int numIslands(char[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    res += 1;
                    numIslandsDfs(grid, i, j);
                }
            }
        }
        return res;
    }

    private static void numIslandsDfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '0';
        numIslandsDfs(grid, i - 1, j);
        numIslandsDfs(grid, i + 1, j);
        numIslandsDfs(grid, i, j - 1);
        numIslandsDfs(grid, i, j + 1);
    }


    /**
     * 接雨水
     *
     * @param height
     * @return
     */
    public static int trap(int[] height) {
        int res = 0;
        int leftMax = 0, rightMax = 0;
        int left = 0, right = height.length - 1;
        while (left < right) {
            leftMax = Math.max(leftMax, height[left]);
            rightMax = Math.max(rightMax, height[right]);
            if (height[left] < height[right]) {
                res += leftMax - height[left];
                left++;
            } else {
                res += rightMax - height[right];
                right--;
            }
        }
        return res;
    }

    /**
     * 合并两个有序数组
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     * @return
     */
    public static void mergeTwoSortedArrays(int[] nums1, int m, int[] nums2, int n) {
        int p1 = m - 1, p2 = n - 1, p = m + n - 1;
        int cur = 0;
        while (p1 >= 0 || p2 >= 0) {
            if(p1==-1){
                nums1[p]=nums2[p2];
                p2--;
                p--;
            }else if(p2==-1){
                nums1[p]=nums1[p1];
                p1--;
                p--;
            }
            else if (nums1[p1] > nums2[p2]) {
                nums1[p] = nums1[p1];
                p--;
                p1--;
            } else {
                nums1[p] = nums2[p2];
                p--;
                p2--;
            }
        }
    }
}








