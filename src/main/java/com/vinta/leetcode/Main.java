package com.vinta.leetcode;


import java.util.function.Function;
import java.util.function.Supplier;

public class Main {

    public static void main(String[] args) {

        Solution solution = new Solution();
        int[][] nums = {{1,2,3},{4,5,6},{7,8,9}};
        int k = 5;
        String s = "abba";
//        SolutionUtils.printResult(solution.firstMissingPositive(nums));
        solution.rotate(nums);
        SolutionUtils.printResult(nums);
    }
}


