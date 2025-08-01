package com.vinta.leetcode;


public class Main {

    public static void main(String[] args) {

        Solution solution = new Solution();
        int[][] nums = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        int[] nums1 = {0};
        int[] nums2 = {};
        int k = 5;
        String s = "abba";
        ListNode list1 = SolutionUtils.createListNode(new Integer[]{1, 2, 3, 4, 5});
        TreeNode treeNode = SolutionUtils.createTreeNode(new Integer[]{3, 9, 20, null, null, 15, 7});
        ListNode list3 = SolutionUtils.createListNode(new Integer[]{2, 6});
        SolutionUtils.printResult(solution.jump(new int[]{2,5,3,2,4,1,0,1}));
//        SolutionUtils.printResult(treeNode);
    }
}


