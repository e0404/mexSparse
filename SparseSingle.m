classdef SparseSingle 
    %< matlab.mixin.indexing.RedefinesParen -> New way of indexing
    properties (SetAccess = private, Hidden = true)
        objectHandle; % Handle to the underlying C++ class instance
        cleanup;
    end

    properties (SetAccess = private)
        nRows
        nCols
    end

    methods
        %% Constructor - Create a new C++ class instance 
        function this = SparseSingle(varargin)
            if nargin == 1 && isa(varargin{1},'uint64')
                this.objectHandle = varargin{1};
            else                
                this.objectHandle = mexSparseSingle('new', varargin{:});
            end
            this.cleanup = onCleanup(@() delete(this));
        end
        
        %% Destructor - Destroy the C++ class instance
        function delete(this)
            mexSparseSingle('delete', this.objectHandle);
        end

        %% nnz
        function nnz = nnz(this)
            nnz = mexSparseSingle('nnz',this.objectHandle);
        end

        %% size
        function sz = size(this)
            sz = mexSparseSingle('size',this.objectHandle);
        end

        function ret = horzcat(varargin)
            error('sparseSingle:missingImplementation','horzcat not yet implemented!');
        end
        
        %% Plus & Minus
        function ret = plus(A,B)
            %error('sparseSingle:missingImplementation','plus operator not yet implemented!');
            if isempty(A) || isempty(B)
                error('Arrays have incompatible sizes for this operation');            

            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                ret = mexSparseSingle('plus',A.objectHandle,B);               
                % matrix * vector
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                ret = mexSparseSingle('plus',B.objectHandle,A);      
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                ret = SparseSingle(mexSparseSingle('plus',A.objectHandle,B.objectHandle));      
            else
                error('Inputs %s & %s not supported', class(A),class(B));
            end
        end

        function ret = times(A,B)
            if isempty(A) || isempty(B)
                error('Arrays have incompatible sizes for this operation');            

            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                ret = mexSparseSingle('times',A.objectHandle,B);               
                % matrix * vector
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                ret = mexSparseSingle('times',B.objectHandle,A);      
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                ret = mexSparseSingle('times',A.objectHandle,B.objectHandle);      
            else
                error('Inputs %s & %s not supported', class(A),class(B));
            end

            ret = SparseSingle(ret);
        end

        function ret = minus(A,B)
            if isempty(A) || isempty(B)
                error('Arrays have incompatible sizes for this operation');            

            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                ret = mexSparseSingle('minusAsMinuend',A.objectHandle,B);               
                % matrix * vector
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                ret = mexSparseSingle('minusAsSubtrahend',B.objectHandle,A);      
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                ret = SparseSingle(mexSparseSingle('minusAsMinuend',A.objectHandle,B.objectHandle));      
            else
                error('Inputs %s & %s not supported', class(A),class(B));
            end
        end

        function ret = uminus(this)            
            ret = SparseSingle(mexSparseSingle('uminus',this.objectHandle));
        end

        function ret = ldivide(A,B)            
            warning('rdivide for sparseSingle behavior differs from its double counterpart, as only nnzs are affected!');
            if isempty(A) || isempty(B)
                error('Arrays have incompatible sizes for this operation');            
            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                retHandle = mexSparseSingle('ldivide',A.objectHandle,B);               
                % matrix * vector
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                retHandle = mexSparseSingle('rdivide',B.objectHandle,A);      
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                retHandle = mexSparseSingle('ldivide',A.objectHandle,B.objectHandle);      
            else
                error('Inputs %s & %s not supported', class(A),class(B));
            end

            ret = SparseSingle(retHandle);
        end

        function ret = rdivide(A,B)     
            warning('rdivide for sparseSingle behavior differs from its double counterpart, as only nnzs are affected!');
            if isempty(A) || isempty(B)
                error('Arrays have incompatible sizes for this operation');            
            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                retHandle = mexSparseSingle('rdivide',A.objectHandle,B);               
                % matrix * vector
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                retHandle = mexSparseSingle('ldivide',B.objectHandle,A);      
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                retHandle = mexSparseSingle('rdivide',A.objectHandle,B.objectHandle);      
            else
                error('Inputs %s & %s not supported', class(A),class(B));
            end

            ret = SparseSingle(retHandle);
        end

        function ret = mldivide(A,B)            
            if (issparse(A) && isnumeric(B) && ~issparse(B))
                ret = mexSparseSingle('mldivide',A.objectHandle,B);
            elseif issparse(A) && issparse(B)
                ret = SparseSingle(mexSparseSingle('mldivide',A.objectHandle,B.objectHandle));
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                ret = A\full(B);
            else
                error('Inputs %s & %s not supported', class(A),class(B));  
            end
        end

        function ret = mrdivide(A,B)            
            error('sparseSingle:missingImplementation','rdivide not yet implemented!');
        end

        function ret = find(this)            
            ret = mexSparseSingle('find',this.objectHandle);
        end
        
        %% sparsity oprations
        function ret = full(this)
            ret = mexSparseSingle('full',this.objectHandle);
        end

        function ret = issparse(this)
            ret = true;
        end

        function ret = isa(this,typeStr)
           ret = false;
           if strcmpi(typeStr,'numeric') || strcmpi(typeStr,'SparseSingle') || strcmpi(typeStr,'single')
              ret = true;
           end
        end

        function ret = isnumeric(this)
           ret = true;
        end

        %% disp
        function disp(this)
            mexSparseSingle('disp',this.objectHandle);
        end

        %%mtimes
        function ret = mtimes(A,B)
            if isempty(A) || isempty(B)
                error('One Input is empty');

                % vector * matrix
                % arg1: numeric vector
                % arg2: SparseSingle obj
            elseif isa(A, 'SparseSingle') && isnumeric(B) && ~issparse(B)
                ret = mexSparseSingle('mtimesr',A.objectHandle,B); 
                % matrix * vector     
            elseif isa(A, 'SparseSingle') && isa(B,'SparseSingle')
                ret = SparseSingle(mexSparseSingle('mtimesr',A.objectHandle,B.objectHandle));  
                 
            elseif isa(B, 'SparseSingle') && isnumeric(A) && ~issparse(A)
                ret = mexSparseSingle('mtimesl',B.objectHandle,A);
                if isscalar(A)              
                    ret = SparseSingle(ret);
                end
            %{
            elseif isrow(A) && isa(B, 'SparseSingle')

                if ~isnumeric(A)
                    error('First Input (arg1) must be numeric');
                end

                if ~isa(A, 'single')
                    A = single(A);
                end
                
                if isscalar(A)
                    ret = SparseSingle(mexSparseSingle('timesScalar',B.objectHandle,A));
                elseif numel(A) == B.nRows
                    ret = mexSparseSingle('vecTimes',B.objectHandle,A);
                else
                    error('Invalid Dimensions for multiplication!');
                end

                % matrix * vector
            elseif isa(A, 'SparseSingle') && iscolumn(B)

                if ~isnumeric(B)
                    error('Second Input (arg2) must be numeric');
                end

                if ~isa(B, 'single')
                    B = single(B);
                end

                if isscalar(B)
                    ret = SparseSingle(mexSparseSingle('timesScalar',A.objectHandle,B));
                elseif numel(B) == A.nCols
                    ret = mexSparseSingle('timesVec',A.objectHandle,B);
                else
                    error('Invalid Dimensions for multiplication!');
                end

            elseif ismatrix(A) && ismatrix(B)
                error('Matrix Matrix product not implemented');
            %}
            else
                error('mtimes for inputs %s & %s not supported', class(A),class(B));
            end
        end

        function ret = transpose(this)
            ret = SparseSingle(mexSparseSingle('transpose',this.objectHandle));
        end

        function ret = ctranspose(this)
            %Todo: difference between conjugate and non-conjugate transpose
            ret = SparseSingle(mexSparseSingle('transpose',this.objectHandle));
        end
        
        %%getter functions
        function nr = get.nRows(this)
            nr = size(this);
            nr = nr(1);
        end

        function nr = get.nCols(this)
            nr = size(this);
            nr = nr(2);
        end

        %% Indexing        
        function values = subsref(this,s)
            switch s(1).type
                case '()'
                    if length(s) == 1
                        nSubs = length(s.subs);

                        %Linear Indexing
                        if nSubs == 1
                            subMatrixHandle = mexSparseSingle('linearIndexing',this.objectHandle,s.subs{1});
                            values = SparseSingle(subMatrixHandle);

                        %Submatrix indexing
                        elseif nSubs == 2
                            %Workaround for Colon at the moment
                            if isequal(s.subs{1},':')
                                warning('Row Colon indexing not efficient at the moment!');
                                s.subs{1} = 1:this.nRows;
                            end

                            if isequal(s.subs{2},':')
                                warning('Column Colon indexing not efficient at the moment!');
                                s.subs{2} = 1:this.nCols;
                            end

                            % Workaround for Logical indexing at the moment

                            if islogical(s.subs{1})
                                if ~isvector(s.subs{1}) || numel(s.subs{1}) ~= this.nRows
                                    error('Wrong index dimension: Number of elements must be the same!');
                                end
                                warning('Logical indexing not efficient at the moment and will be converted to an index list!');
                                s.subs{1} = find(s.subs{1});
                            end

                            if islogical(s.subs{2})
                                if ~isvector(s.subs{2}) || numel(s.subs{1}) ~= this.nCols
                                    error('Wrong index dimension: Number of elements must be the same!');
                                end
                                warning('Logical indexing not efficient at the moment and will be converted to an index list!');
                                s.subs{2} = find(s.subs{2});
                            end

                            subMatrixHandle = mexSparseSingle('subsrefRowCol',this.objectHandle,s.subs{1},s.subs{2});
                            values = SparseSingle(subMatrixHandle);
                        else
                            error('Requested Indexing Pattern is not supported!');
                        end
                        
                    elseif length(s) >= 2 && strcmp(s(2).type,'.')
                        error('Dot indexing not supported for SparseSingle!');
                    else
                        error('Requested indexing pattern not supported!');
                    end
                case '.'
                    [varargout{1:nargout}] = builtin('subsref',this,s);
                case '{}'
                    error('{} indexing not supported for SparseSingle!');
                otherwise
                    error('Not a valid indexing expression');
            end
        end
    end
end