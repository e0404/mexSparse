classdef SparseSingleTest < matlab.unittest.TestCase

    methods (Test)
        function test_constructBasics(testCase)
            test = sparse(eye(2));
            tests = SparseSingle(test);
            testCase.fatalAssertClass(tests,'SparseSingle');
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertTrue(isnumeric(tests));
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));
            testCase.fatalAssertEqual(nnz(test),nnz(tests));
            testCase.fatalAssertEqual(size(test),size(tests));
        end

        function test_transpose(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test);

            testt = test';
            testst = tests';
            testCase.fatalAssertEqual(size(testt),size(testst));            
        end

        function test_full(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test);

            testf = full(test);
            testsf = full(tests);
            testCase.verifyEqual(size(testf),size(testsf));
            testCase.verifyEqual(norm(testf,'fro'),double(norm(testsf,'fro')),'relTol',1e-5);

            testf = full(test');
            testsf = full(tests');
            testCase.verifyEqual(size(testf),size(testsf));

        end

        function test_mtimes_Ax(testCase)
            test = sparse(eye(2));
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test*v;
            prod2 = hello*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_mtimes_Atx(testCase)
            test = sparse(eye(2));
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test'*v;
            prod2 = hello'*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_mtimes_xA(testCase)
            test = sparse([0 2 0; 1 0 0]);
            hello = SparseSingle(test);
            v = [1 1];
            vs = single(v);
            prod1 = v*test;
            prod2 = vs*hello;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_mtimes_xAt(testCase)
            test = sparse([0 2 0; 1 0 0])';
            hello = SparseSingle(test);
            v = [1 1];
            vs = single(v);
            prod1 = v*test';
            prod2 = vs*hello';

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_transposed(testCase)
            test = sparse([0 2 0; 1 0 0]);
            tests = SparseSingle(test);
            tests_t = transpose(tests);

            prod1 = tests*[1;1;1];
            prod2 = tests_t'*[1;1;1];

            testCase.verifySize(tests_t,fliplr(size(tests))); 
            testCase.verifyTrue(norm(prod1 - prod2) < eps('single'));
        end

        function test_ScalarMultiply(testCase)
            test = sparse([0 2 0; 1 0 0]);
            tests = SparseSingle(test);
            test_t = transpose(test);
            tests_t = transpose(tests);
            
            result = 2*test;
            result_t = 2*test_t;
            result_s = 2*tests;
            result_t_s = 2*tests_t;

            testCase.verifySize(result_s,size(result_t));
            testCase.verifySize(result_t_s,size(result_t));             
        end

        function test_subsref_blocks(testCase)
            test = sprand(10,10,0.1);
            tests = SparseSingle(test);
            
            block{1} = test(:,:);
            blocks{1} = tests(:,:);

            block{2} = test(:,5);
            blocks{2} = tests(:,5);

            block{3} = test(5,:);
            blocks{3} = tests(5,:);

            block{4} = test(2:7,:);
            blocks{4} = tests(2:7,:);

            block{5} = test(:,2:7);
            blocks{5} = tests(:,2:7);

            block{6} = test(2:7,2:7);
            blocks{6} = tests(2:7,2:7);

            block{5} = test(3,2:7);
            blocks{5} = tests(3,2:7);

            block{6} = test(2:7,3);
            blocks{6} = tests(2:7,3);
            
            for i = 1:numel(block)
                testCase.verifySize(blocks{i},size(block{i}));
                testCase.verifyEqual(nnz(blocks{i}),nnz(block{i}));
            end
        end

        function test_subsref_slicing(testCase)
            test = sprand(10,10,0.1);
            tests = SparseSingle(test);
            
            slice{1} = test([2 6],:);
            slices{1} = tests([2 6],:);

            slice{2} = test(:,[2 6]);
            slices{2} = tests(:,[2 6]);

            slice{3} = test([2 7],[2 6]);
            slices{3} = tests([2,7],[2 6]);

            slice{4} = test([2 7 4],[2 6]);
            slices{4} = tests([2 7 4],[2 6]);

            slice{5} = test([2 6],[2 7 4]);
            slices{5} = tests([2 6],[2 7 4]);
            
            for i = 1:numel(slice)
                testCase.verifySize(slices{i},size(slice{i}));
                testCase.verifyEqual(nnz(slices{i}),nnz(slice{i}));
            end
        end

        function test_linearindexingColon(testCase)
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
            testCase.verifyTrue(all((full(ixVecD) - full(ixVecS)) < eps('single')));
            testCase.verifyTrue(all((full(ixVecDT) - full(ixVecST)) < eps('single')));
        end
    end

end


