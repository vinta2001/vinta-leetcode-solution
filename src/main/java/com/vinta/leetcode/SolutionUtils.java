package com.vinta.leetcode;

import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.Queue;

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
    public static TreeNode createTreeNode(Integer[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return createTreeNodeHelper(nums, 0);
    }

    private static TreeNode createTreeNodeHelper(Integer[] nums, int index) {
        if (index >= nums.length || nums[index]==null) {
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
    public static ListNode createListNode(Integer[] nums) {
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

    // 打印二叉树
    public static void printResult(TreeNode root) {
        if (root == null) {
            System.out.println("空树");
            return;
        }

        // 获取树的高度
        int height = getHeight(root);
        // 计算最大宽度（每一层的最大字符数）
        int maxWidth = (int) Math.pow(2, height) * 2 - 3;

        // 使用队列进行层次遍历，同时存储节点所在的位置
        Queue<NodePosition> queue = new LinkedList<>();
        // 根节点初始位置在中间
        queue.add(new NodePosition(root, 0, maxWidth / 2));

        int currentLevel = 1;
        int nodesInLevel = 1;
        int printedInLevel = 0;

        // 记录上一层节点的位置，用于绘制连接线
        int[] prevPositions = new int[(int) Math.pow(2, height - 1)];
        int prevIndex = 0;

        while (!queue.isEmpty() && currentLevel <= height) {
            NodePosition np = queue.poll();
            TreeNode node = np.node;
            int level = np.level;
            int pos = np.position;

            // 如果是新的一层，打印换行
            if (printedInLevel == 0) {
                System.out.println();
            }

            // 打印当前节点前的空格
            int spaces = (printedInLevel == 0) ? pos : pos - getLastPosition(prevPositions, printedInLevel) - 1;
            printSpaces(spaces);

            // 打印节点值
            if (node != null) {
                System.out.print(node.val);
                prevPositions[printedInLevel] = pos;

                // 添加子节点到队列
                int levelDiff = height - level - 1;
                int childSpacing = (levelDiff > 0) ? (int) Math.pow(2, levelDiff) : 1;

                queue.add(new NodePosition(node.left, level + 1, pos - childSpacing));
                queue.add(new NodePosition(node.right, level + 1, pos + childSpacing));
            } else {
                System.out.print(" ");
                // 为 null 节点添加占位的子节点
                queue.add(new NodePosition(null, level + 1, 0));
                queue.add(new NodePosition(null, level + 1, 0));
            }

            printedInLevel++;

            // 如果当前层打印完毕
            if (printedInLevel == nodesInLevel) {
                // 打印连接线（除了最后一层）
                if (currentLevel < height) {
                    printConnectors(prevPositions, nodesInLevel, currentLevel, height);
                }

                // 准备下一层
                currentLevel++;
                nodesInLevel *= 2;
                printedInLevel = 0;
                prevIndex = 0;
            }
        }
        System.out.println();
    }

    // 打印节点之间的连接线
    private static void printConnectors(int[] positions, int count, int level, int height) {
        System.out.println();
        int levelDiff = height - level - 1;
        int connectorSpacing = (levelDiff > 0) ? (int) Math.pow(2, levelDiff) : 1;

        for (int i = 0; i < count; i++) {
            int pos = positions[i];
            if (pos == 0) continue; // 跳过空节点

            // 打印左连接线前的空格
            int leftSpace = (i == 0) ? (pos - connectorSpacing) :
                    (pos - connectorSpacing - getLastPosition(positions, i) - 1);
            printSpaces(leftSpace);
            System.out.print("/");

            // 打印左右连接线之间的空格
            printSpaces(2 * connectorSpacing - 1);
            System.out.print("\\");
        }
    }

    // 获取上一个已打印节点的位置
    private static int getLastPosition(int[] positions, int currentIndex) {
        if (currentIndex == 0) return 0;
        for (int i = currentIndex - 1; i >= 0; i--) {
            if (positions[i] != 0) {
                return positions[i];
            }
        }
        return 0;
    }

    // 打印指定数量的空格
    private static void printSpaces(int count) {
        for (int i = 0; i < count; i++) {
            System.out.print(" ");
        }
    }

    // 获取树的高度
    private static int getHeight(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(getHeight(root.left), getHeight(root.right));
    }

    // 辅助类：存储节点及其位置信息
    static class NodePosition {
        TreeNode node;
        int level;
        int position;

        NodePosition(TreeNode node, int level, int position) {
            this.node = node;
            this.level = level;
            this.position = position;
        }
    }
}
