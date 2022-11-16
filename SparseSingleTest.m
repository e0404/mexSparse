classdef SparseSingleTest < matlab.unittest.TestCase
    
    properties (Constant)
        relTolerance = 5*eps('single');
    end

    methods (Test)
        function test_constructFromSparseDouble(testCase)
            test = sparse(eye(2));
            tests = SparseSingle(test);
            testCase.fatalAssertClass(tests,'SparseSingle');
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertTrue(isnumeric(tests));
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));
            testCase.fatalAssertEqual(nnz(test),nnz(tests));
            testCase.fatalAssertEqual(size(test),size(tests));
        end

        function test_constructFromDense(testCase)
            %Add tests for other constructors
            tests = SparseSingle(eye(5));
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertSize(tests,[5 5]);
            testCase.fatalAssertEqual(nnz(tests),5);
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));

            tests = SparseSingle(single(eye(5)));
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertSize(tests,[5 5]);
            testCase.fatalAssertEqual(nnz(tests),5);
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));

            testCase.verifyError(@() SparseSingle("hello"),'sparseSingle:invalidInputType');
        end

        function test_constructFromSize(testCase)

            tests = SparseSingle(3,4);
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertSize(tests,[3 4]);
            testCase.fatalAssertEqual(nnz(tests),0);
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));            

            %Invalid Inputs
            testCase.verifyError(@() SparseSingle([3 2],4),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle([3 2],[2 3]),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle(3,[2 3]),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle("hello",1),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle(1,"hello"),'sparseSingle:invalidInputType');
            
            %You can also construct sparse doubles from rows/columns given as char, uint, etc. in Matlab. Should we test that?
        end

        function test_constructFromTriplets(testCase)
            i = 1:4; j = 2:5; v = rand(4,1,'single');
            tests = SparseSingle(i,j,v);
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertSize(tests,[4 5]);
            testCase.fatalAssertEqual(nnz(tests),4);
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));   

            tests = SparseSingle(i,j,v,10,20);
            testCase.fatalAssertTrue(issparse(tests));
            testCase.fatalAssertSize(tests,[10 20]);
            testCase.fatalAssertEqual(nnz(tests),4);
            testCase.fatalAssertTrue(isa(tests,'numeric') && isa(tests,'single'));  

            %Invalid Inputs
            testCase.verifyError(@() SparseSingle([3 2],4,ones(1,2,'single')),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle([3 2],[2 3],ones(1,2,'double')),'sparseSingle:invalidInputType');
            testCase.verifyError(@() SparseSingle([3 2],"test",ones(1,2,'single')),'sparseSingle:invalidDataPointer');
        end

        function test_horzcat(testCase)
            test = sprand(50,50,0.05);
            tests = SparseSingle(test);

            test = [test test];
            tests = [tests tests];
            
            testCase.verifySize(tests,size(test));
            testCase.verifyEqual(nnz(test),nnz(tests));
        end

        function test_transpose(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test);

            testt = test';
            testst = tests';
            testCase.fatalAssertEqual(size(testt),size(testst));            
        end

        function test_find(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test);
            
            testCase.verifyEqual(find(test),find(tests));
            testCase.verifyEqual(find(test'),find(tests'));
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

        function test_plus(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test); 
            
            %Scalar Addition
            test5 = test + 5;
            tests5 = tests + 5;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(test5 - tests5 < testCase.relTolerance*test5,'all'));            

            tests5 = 5 + tests;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(test5 - tests5 < testCase.relTolerance*test5,'all'));
            
            
            %Matrix addition
            testM = test + ones(size(test));
            testsM = tests + ones(size(test),'single');
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(testM - testsM < testCase.relTolerance*testM,'all'));
            
            testsM = ones(size(test),'single') + tests;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(testM - testsM < testCase.relTolerance*testM,'all'));

            %Sparse sparse
            testsM = tests + tests;
            testM = test + test;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= testCase.relTolerance*full(testM),'all'));
            testCase.verifyEqual(nnz(testsM),nnz(testM));

            %Wrong size error
            testCase.verifyError(@() ones(3) + tests,'sparseSingle:wrongDataType');
            testCase.verifyError(@() tests + ones(3),'sparseSingle:wrongDataType');
        end

        function test_minus(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test); 
            
            %Scalar Addition
            test5 = test - 5;
            tests5 = tests - 5;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(test5 - tests5) <= abs(testCase.relTolerance*test5),'all'));            

            tests5 = 5 - tests;
            test5 = 5 - test;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(test5 - tests5) <= abs(testCase.relTolerance*test5),'all'));
            
            
            %Matrix addition
            testM = test - ones(size(test));
            testsM = tests - ones(size(test),'single');
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            compare = abs(testM - testsM) <= abs(testCase.relTolerance*testM);
            testCase.verifyTrue(all(compare,'all'));
            
            testM = ones(size(test)) - test;
            testsM = ones(size(test),'single') - tests;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            compare = abs(testM - testsM) <= abs(testCase.relTolerance*testM);
            testCase.verifyTrue(all(compare,'all'));
            
            %Sparse sparse
            testsM = tests - tests;
            testM = test - test;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            testCase.verifyTrue(nnz(testsM) == 0);

            %Wrong size error
            testCase.verifyError(@() ones(3) + tests,'sparseSingle:wrongDataType');
            testCase.verifyError(@() tests + ones(3),'sparseSingle:wrongDataType');
        end

        function test_times(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test); 
            
            %Scalar Addition
            test5 = test .* 5;
            tests5 = tests .* 5;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     

            tests5 = 5 .* tests;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     
            
            
            %Matrix addition
            testM = test .* ones(size(test));
            testsM = tests .* ones(size(test),'single');
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            
            testsM = ones(size(test),'single') .* tests;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));

            %Sparse sparse
            testsM = tests .* tests;
            testM = test .* test;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            testCase.verifyEqual(nnz(testsM),nnz(testM));

            %Wrong size error
            testCase.verifyError(@() ones(3) + tests,'sparseSingle:wrongDataType');
            testCase.verifyError(@() tests + ones(3),'sparseSingle:wrongDataType');
        end

        function test_rdivide(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test); 
            
            %Scalar Addition
            test5 = test ./ 5;
            tests5 = tests ./ 5;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     

            tests5 = 5 ./ tests;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     
            
            
            %Matrix addition
            testM = test ./ ones(size(test));
            testsM = tests ./ ones(size(test),'single');
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            
            testsM = ones(size(test),'single') .* tests;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));

            %Sparse sparse
            testsM = tests ./ tests;
            testM = test ./ test;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            testCase.verifyEqual(nnz(testsM),nnz(testM));

            %Wrong size error
            testCase.verifyError(@() ones(3) + tests,'sparseSingle:wrongDataType');
            testCase.verifyError(@() tests + ones(3),'sparseSingle:wrongDataType');
        end

        function test_ldivide(testCase)
            test = sprand(5,10,0.25);
            tests = SparseSingle(test); 
            
            %Scalar Addition
            test5 = test .\ 5;
            tests5 = tests .\ 5;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     

            tests5 = 5 .\ tests;
            testCase.assertEqual(issparse(test5),issparse(tests5)); %Equal behavior
            testCase.verifySize(tests5,size(test5));
            testCase.verifyTrue(all(abs(full(test5) - full(tests5)) <= abs(testCase.relTolerance*full(test5)),'all'));     
            
            
            %Matrix addition
            testM = test .\ ones(size(test));
            testsM = tests .\ ones(size(test),'single');
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            
            testsM = ones(size(test),'single') .* tests;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));

            %Sparse sparse
            testsM = tests .\ tests;
            testM = test .\ test;
            testCase.assertEqual(issparse(testM),issparse(testsM)); %Equal behavior
            testCase.verifySize(testsM,size(testM));
            testCase.verifyTrue(all(abs(full(testM) - full(testsM)) <= abs(testCase.relTolerance*full(testM)),'all'));
            testCase.verifyEqual(nnz(testsM),nnz(testM));

            %Wrong size error
            testCase.verifyError(@() ones(3) + tests,'sparseSingle:wrongDataType');
            testCase.verifyError(@() tests + ones(3),'sparseSingle:wrongDataType');
        end
            
        function test_uminus(testCase)
            test = sprand(10,10,0.1);
            tests = -SparseSingle(test);
            
            testCase.assertEqual(nnz(test),nnz(tests));
            testCase.assertSize(tests,size(tests));
            testCase.verifyTrue(all(abs(full(tests) + full(test)) <= abs(testCase.relTolerance*full(test)),'all'));            
        end

        function test_mtimes_Ax(testCase)
            test = sparse(eye(2));
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test*v;
            prod2 = hello*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < testCase.relTolerance);
        end

        function test_mtimes_Atx(testCase)
            test = sparse(eye(2));
            hello = SparseSingle(test);
            v = [1; 1];
            vs = single(v);
            prod1 = test'*v;
            prod2 = hello'*v;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < testCase.relTolerance);
        end

        function test_mtimes_xA(testCase)
            test = sparse([0 2 0; 1 0 0]);
            hello = SparseSingle(test);
            v = [1 1];
            vs = single(v);
            prod1 = v*test;
            prod2 = vs*hello;

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < testCase.relTolerance);
        end

        function test_mtimes_xAt(testCase)
            test = sparse([0 2 0; 1 0 0])';
            hello = SparseSingle(test);
            v = [1 1];
            vs = single(v);
            prod1 = v*test';
            prod2 = vs*hello';

            testCase.verifySize(prod2,size(prod1));
            testCase.verifyTrue(norm(prod1 - prod2) < testCase.relTolerance);
        end

        function test_transposed(testCase)
            test = sparse([0 2 0; 1 0 0]);
            tests = SparseSingle(test);
            tests_t = transpose(tests);

            prod1 = tests*[1;1;1];
            prod2 = tests_t'*[1;1;1];

            testCase.verifySize(tests_t,fliplr(size(tests))); 
            testCase.verifyTrue(norm(prod1 - prod2) < testCase.relTolerance);
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

            testCase.verifySize(result_s,size(result));
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
            testCase.verifyTrue(all((full(ixVecD) - full(ixVecS)) < testCase.relTolerance));
            testCase.verifyTrue(all((full(ixVecDT) - full(ixVecST)) < testCase.relTolerance));
        end
    end

end


