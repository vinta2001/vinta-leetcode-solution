package com.vinta.leetcode;

import java.util.Arrays;
import java.util.Collection;

public class SolutionUtils {
    public static void printResult(int[][] nums) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < nums.length; i++) {
            sb.append(Arrays.toString(nums[i]));
            if (i != nums.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        System.out.println(sb);
    }

    public static void printResult(int[] nums) {
        System.out.println(Arrays.toString(nums));
    }

    public static void printResult(String s) {
        System.out.println(s);
    }

    public static void printResult(char[] chars) {
        System.out.println(chars);
    }

    public static <T> void printResult(Collection<T> collection) {
        System.out.println(collection);
    }

    public static void printResult(int num) {
        System.out.println(num);
    }

    /**
     * 根据数组生成二叉树
     *
     * @param nums
     * @return 二叉树
     */
    public static TreeNode createTreeNode(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return createTreeNodeHelper(nums, 0);
    }

    private static TreeNode createTreeNodeHelper(int[] nums, int index) {
        if (index >= nums.length || nums[index] == -1) {
            return null;
        }
        TreeNode node = new TreeNode(nums[index]);
        node.left = createTreeNodeHelper(nums, 2 * index + 1);
        node.right = createTreeNodeHelper(nums, 2 * index + 2);
        return node;
    }


    /**
     * 根据数组生成链表
     *
     * @param nums
     * @return 链表
     */
    public static ListNode createListNode(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        for (int num : nums) {
            current.next = new ListNode(num);
            current = current.next;
        }
        return dummy.next;
    }

    public static void printResult(ListNode head) {
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        while (head != null) {
            sb.append(head.val).append(",");
            head = head.next;
        }
        sb.deleteCharAt(sb.length() - 1);
        sb.append(']');
        System.out.println(sb);
    }
}
