classdef SparseSingleTest < matlab.unittest.TestCase

    methods (Test)
        function test_mtimes_1(testCase)
            test = sparse(eye(2))
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test*v;
            prod2 = hello*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_mtimes_2(testCase)
            test = sparse(eye(2))
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test'*v;
            prod2 = hello'*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_transposed(testCase)
            test = sparse([0 2 0; 1 0 0]);
            tests = SparseSingle(test);
            tests_t = transpose(tests);

            prod1 = tests*[1;1];
            prod2 = tests_t'*[1;1];

            testCase.verifySize(tests_t,fliplr(size(tests))); 
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_linearindexing(testCase)
            test = sparse([0 2 0; 1 0 0]);
            tests = SparseSingle(test);
            test_t = transpose(test);
            tests_t = transpose(tests);
            
            ixVecD = test(:);
            ixVecS = tests(:);
            ixVecDT = test_t(:);
            ixVecST = tests_t(:);

            testCase.verifySize(ixVecS,size(ixVecD));
            testCase.verifySize(ixVecST,size(ixVecDT)); 
            testCase.verifyTrue(all((full(ixVecD) - ixVecS) < eps('single')));
            testCase.verifyTrue(all((full(ixVecDT) - ixVecST) < eps('single')));
        end
    end

end


