-module(bank).
%%-compile(export_all).
-export([create_bank/0,new_account/2,withdraw_money/3,deposit_money/3,transfer/4,balance/2,terminate/1]).

%% Client API
create_bank() -> spawn(fun() -> loop([]) end).

new_account(Pid,AccountNumber) ->
	Pid ! {new_account,AccountNumber,self()},
	receive
		Msg -> Msg
	end.

withdraw_money(Pid,AccountNumber,Quantity) ->
	Pid ! {withdraw_money,AccountNumber,Quantity,self()},
	receive
		Msg -> Msg
	end.

deposit_money(Pid, AccountNumber,Quantity) ->
	Pid ! {deposit_money,AccountNumber,Quantity,self()},
	receive
		Msg -> Msg
	end.

transfer(Pid,FromAccount,ToAccount,Quantity) ->
	Pid ! {transfer,FromAccount,ToAccount,Quantity,self()},
	receive
		Msg -> Msg
	end.

balance(Pid,Account) ->
	Pid ! {balance,Account,self()},
	receive
		Msg -> Msg
	end.

terminate(Pid) ->
	Pid ! {terminate,self()},
	receive
		Msg -> Msg
	end.

loop(Accounts) ->
	receive
		{new_account,AccountNumber,From} ->
			case proplists:lookup(AccountNumber,Accounts) of
				none -> %% Podemos crear esa cuenta
					From ! true,
					loop([{AccountNumber,0}|Accounts]);
				{AccountNumber,_} -> %% Ya existe esa cuenta
					From ! false,
					loop(Accounts)
			end;

		{withdraw_money,AccountNumber,Quantity,From} ->
			case check_balance(AccountNumber,Accounts) of
				none -> %% No existe la cuenta
					From ! {error, not_found},
					loop(Accounts);
				Balance when Quantity =< Balance->
					From ! {ok,Quantity},
					NewAccounts = lists:keyreplace(AccountNumber,1,Accounts,{AccountNumber,Balance-Quantity}),
					loop(NewAccounts);
				Balance when Quantity > Balance ->
					From ! {error, not_money},
					loop(Accounts)
			end;

		{deposit_money,AccountNumber,Quantity,From} ->
			case proplists:lookup(AccountNumber,Accounts) of
				none -> %% No existe la cuenta
					From ! {error, not_found},
					loop(Accounts);
				{AccountNumber,_} ->
					Balance = proplists:get_value(AccountNumber,Accounts),
					NewBalance = Balance + Quantity,
					From ! {ok,NewBalance},
					NewAccounts = lists:keyreplace(AccountNumber,1,Accounts,{AccountNumber,NewBalance}),
					loop(NewAccounts)
			end;

		{transfer,FromAccount,ToAccount,Quantity,From} ->
			case check_balance(ToAccount,Accounts) of
				none ->
					From ! {error, to_account_not_found},
					loop(Accounts);
				ToBalance ->
					case check_balance(FromAccount,Accounts) of
						none ->
							From ! {error, from_account_not_found},
							loop(Accounts);
						FromBalance when Quantity =< FromBalance ->
							NewFromBalance = FromBalance - Quantity,
							NewToBalance = ToBalance + Quantity,
							From ! {ok,Quantity},
							NewAccounts = lists:keyreplace(FromAccount,1,lists:keyreplace(ToAccount,1,Accounts,{ToAccount,NewToBalance}),{FromAccount,NewFromBalance}),
							loop(NewAccounts);
						FromBalance when Quantity > FromBalance ->
							From ! {error, from_account_not_money},
							loop(Accounts)
					end
			end;

		{balance,Account,From} ->
			case proplists:lookup(Account,Accounts) of
				none ->
					From ! {error,not_found},
					loop(Accounts);
				{Account,_} ->
					Balance = proplists:get_value(Account,Accounts),
					From ! {your_balance, Balance},
					loop(Accounts)
			end;
		{terminate, From} ->
			From ! ok,
			ok;
		Unknown ->
			io:format("Unexpected message: ~p~n",[Unknown]),
			loop(Accounts)
	end.

%% Private functions
check_balance(_,[]) -> none;
check_balance(AccountNumber,[{AccountNumber,Balance}|_]) -> Balance;
check_balance(AccountNumber,[_|T]) -> check_balance(AccountNumber,T).