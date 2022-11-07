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

        %% disp
        function disp(this)
            mexSparseSingle('disp',this.objectHandle);
        end

        %%mtimes
        function ret = mtimes(arg1,arg2)
            if isempty(arg1) || isempty(arg2)
                error('One Input is empty');

                % vector * matrix
                % arg1: numeric vector
                % arg2: SparseSingle obj              

            elseif isrow(arg1) && isa(arg2, 'SparseSingle')

                if ~isnumeric(arg1)
                    error('First Input (arg1) must be numeric');
                end

                if ~isa(arg1, 'single')
                    arg1 = single(arg1);
                end
                
                if isscalar(arg1)
                    ret = mexSparseSingle('timesScalar',arg2.objectHandle,arg1);
                elseif numel(arg1) == arg2.nRows
                    ret = mexSparseSingle('vecTimes',arg2.objectHandle,arg1);
                else
                    error('Invalid Dimensions for multiplication!');
                end

                % matrix * vector
            elseif isa(arg1, 'SparseSingle') && iscolumn(arg2)

                if ~isnumeric(arg2)
                    error('Second Input (arg2) must be numeric');
                end

                if ~isa(arg2, 'single')
                    arg2 = single(arg2);
                end

                if isscalar(arg2)
                    ret = mexSparseSingle('timesScalar',arg1.objectHandle,arg2);
                elseif numel(arg2) == arg1.nCols
                    ret = mexSparseSingle('timesVec',arg1.objectHandle,arg2);
                else
                    error('Invalid Dimensions for multiplication!');
                end

            elseif ismatrix(arg1) && ismatrix(arg2)
                error('Matrix Matrix product not implemented');
            else
                error('Input of type %s not supported', class(v));
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