import os
from web3 import Web3
from datetime import datetime
from typing import NamedTuple, List
import json
import logging
from dotenv import load_dotenv
import time
import json
from typing import Union

# local json DB
class JSONDB:
    def __init__(self, filename='db.json'):
        self.filename = filename
        self._load()

    def _load(self):
        try:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.data = {}

    def _save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def __getitem__(self, key):
        return self.data.get(key, None)

    def __setitem__(self, key, value):
        self.data[key] = value
        self._save()

    def __delitem__(self, key):
        if key in self.data:
            del self.data[key]
            self._save()

    def __contains__(self, key):
        return key in self.data

db = JSONDB()

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
RPC_URL = os.getenv('RPC_URL')
LBP_CONTRACT_ADDRESS = os.getenv('LBP_CONTRACT_ADDRESS')
LBROUTER_CONTRACT_ADDRESS = os.getenv('LBROUTER_CONTRACT_ADDRESS')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
BIN_STEPS = int(os.getenv('BIN_STEPS'))
USE_MAX_FUNDS_TOKENX = os.getenv('USE_MAX_FUNDS_TOKENX')
USE_MAX_FUNDS_TOKENX = int(USE_MAX_FUNDS_TOKENX) if USE_MAX_FUNDS_TOKENX else 0
USE_MAX_FUNDS_TOKENY = os.getenv('USE_MAX_FUNDS_TOKENY')
USE_MAX_FUNDS_TOKENY = int(USE_MAX_FUNDS_TOKENY) if USE_MAX_FUNDS_TOKENY else 0
# New configuration for bin range
BIN_RANGE = int(os.getenv('BIN_RANGE', '10'))  # Default to 10 if not specified

KEY_STORE = os.getenv('KEY_STORE')

POSITION_KEY = f"{KEY_STORE}position"
APPROVED_KEY = f"{KEY_STORE}approved"


class LiquidityPosition(NamedTuple):
    bin_ids: List[int]  # Modified to hold multiple bin IDs
    token_x: str
    token_y: str
    amount_x: int
    amount_y: int
    to_address: str
    range_start: int  # New field to store the start of the bin range
    range_end: int    # New field to store the end of the bin range


class LiquidityManager:

    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        self.account = self.w3.eth.account.from_key(PRIVATE_KEY)

        # Load contract ABIs
        with open('abi/lbp_contract_abi.json', 'r') as f:
            lpb_abi = json.load(f)
        with open('abi/lbrouter_contract_abi.json', 'r') as f:
            lbrouter_abi = json.load(f)
        with open('abi/erc20_contract_abi.json', 'r') as f:
            self.erc20_abi = json.load(f)

        if LBP_CONTRACT_ADDRESS is None:
            raise ValueError(
                "LBP_CONTRACT_ADDRESS environment variable not set.")
        if LBROUTER_CONTRACT_ADDRESS is None:
            raise ValueError(
                "LBROUTER_CONTRACT_ADDRESS environment variable not set.")

        # Initialize contracts
        try:
            self.lbp_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(LBP_CONTRACT_ADDRESS),
                abi=lpb_abi)
            self.lbrouter_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(LBROUTER_CONTRACT_ADDRESS),
                abi=lbrouter_abi)
        except Exception as e:
            logging.error(f"Error initializing contracts: {str(e)}")
            raise

    def get_active_position(self) -> Union[LiquidityPosition, None]:
        """Fetch active liquidity positions from database."""
        if POSITION_KEY not in db:
            logging.info("No active liquidity position found.")
            return None
        pos = db[POSITION_KEY]
        return LiquidityPosition(
            bin_ids=pos['bin_ids'],
            token_x=pos['token_x'],
            token_y=pos['token_y'],
            amount_x=pos['amount_x'],
            amount_y=pos['amount_y'],
            to_address=pos['to_address'],
            range_start=pos['range_start'],
            range_end=pos['range_end']
        )

    def add_new_position(self, position: LiquidityPosition):
        """Add a new liquidity position to the database."""
        db[POSITION_KEY] = {
            'bin_ids': position.bin_ids,
            'token_x': position.token_x,
            'token_y': position.token_y,
            'amount_x': position.amount_x,  # Store as integer to avoid precision issues
            'amount_y': position.amount_y,  # Store as integer to avoid precision issues
            'to_address': position.to_address,
            'range_start': position.range_start,
            'range_end': position.range_end
        }

    def withdraw_liquidity(self, position: LiquidityPosition) -> bool:
        """Withdraw liquidity from all bins in the position."""
        try:
            # For each bin in the position, check if we have liquidity and withdraw if we do
            bin_ids = []
            amounts = []
            
            logging.info(f"Checking withdrawal liquidity for {len(position.bin_ids)} bins")
            for bin_id in position.bin_ids:
                try:
                    amount = self.lbp_contract.functions.balanceOf(
                        self.account.address, bin_id).call()
                    
                    # Log the bin's balance for debugging
                    logging.info(f"Bin {bin_id} has balance: {amount}")
                    
                    # Only add bins with actual liquidity
                    if amount > 0:
                        bin_ids.append(bin_id)
                        amounts.append(amount)
                except Exception as bin_error:
                    logging.error(f"Error checking balance for bin {bin_id}: {str(bin_error)}")
                    # Continue to next bin rather than failing completely
                    continue
                    
            logging.info(f"Found liquidity in {len(bin_ids)} bins out of {len(position.bin_ids)}")
            
            if not bin_ids:
                logging.warning("No liquidity found in any of the position bins.")
                # Even though we didn't withdraw anything, we can still mark this as successful
                # since there's nothing to withdraw
                return True
                    
            # Build transaction to remove liquidity from all bins at once
            logging.info(f"Withdrawing liquidity from {len(bin_ids)} bins")
            
            # For very large number of bins, we might need to split into multiple transactions
            # to avoid hitting gas limits
            max_bins_per_tx = 90  # Adjust based on network conditions
            
            if len(bin_ids) > max_bins_per_tx:
                logging.info(f"Large number of bins detected, splitting into multiple transactions")
                
                for i in range(0, len(bin_ids), max_bins_per_tx):
                    chunk_bin_ids = bin_ids[i:i+max_bins_per_tx]
                    chunk_amounts = amounts[i:i+max_bins_per_tx]
                    
                    if not self._execute_withdraw(position, chunk_bin_ids, chunk_amounts):
                        return False
                    
                return True
            else:
                # Execute single withdrawal for all bins
                return self._execute_withdraw(position, bin_ids, amounts)

        except Exception as e:
            logging.error(f"Error withdrawing liquidity: {str(e)}")
            return False

    def _execute_withdraw(self, position: LiquidityPosition, bin_ids: list, amounts: list) -> bool:
        """Execute the withdrawal transaction for specified bins and amounts"""
        try:
            logging.info(f"Executing withdrawal for {len(bin_ids)} bins")
            
            remove_tx = self.lbrouter_contract.functions.removeLiquidity(
                position.token_x,
                position.token_y,
                BIN_STEPS,
                0,  # Min amount X
                0,  # Min amount Y
                bin_ids,
                amounts,
                self.account.address,  # owner's wallet
                int(datetime.now().timestamp()) + 3600
            ).build_transaction({
                'from': self.account.address,
                'gas': 5000000,  # Increased gas limit for multiple bins
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })

            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                remove_tx, self.account._private_key)
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_tx.raw_transaction)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                logging.info(f"Successfully removed liquidity from {len(bin_ids)} bins")
                return True
            else:
                logging.error(f"Transaction failed with status {receipt.status}")
                return False
                
        except Exception as e:
            logging.error(f"Error executing withdrawal: {str(e)}")
            return False
    
    def check_token_approval(self, token_address: str,
                             spender_address: str) -> tuple:
        """
        Check if a token is approved for the spender address
        Returns (symbol, current_allowance, has_sufficient_allowance)
        """
        try:
            token_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.erc20_abi)

            # Get token info
            logging.info("Getting token info")
            symbol = token_contract.functions.symbol().call()
            decimals = token_contract.functions.decimals().call()
            logging.info("Fetched token info")

            # Check allowance
            logging.info("Checking allowance")
            allowance = token_contract.functions.allowance(
                self.account.address, spender_address).call()

            logging.info("Fetched allowance")
            # Convert allowance to human readable format
            human_allowance = allowance / (10**decimals)

            # Consider it approved if allowance is more than 1M tokens (arbitrary large number)
            # You might want to adjust this threshold based on your needs
            has_sufficient_allowance = allowance > (10**decimals * 1_000_000)

            return symbol, human_allowance, has_sufficient_allowance

        except Exception as e:
            logging.error(
                f"Error checking approval for token {token_address}: {str(e)}")
            return None, 0, False

    def get_token_balance(self, token_address: str) -> tuple:
        """Get balance and info for a specific token."""
        try:
            token_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.erc20_abi)

            logging.info("Fetching token symbol and balance")
            symbol = token_contract.functions.symbol().call()
            balance = token_contract.functions.balanceOf(
                self.account.address).call()

            logging.info("Fetched token symbol and balance")
            return symbol, balance
        except Exception as e:
            logging.error(
                f"Error getting token balance for {token_address}: {str(e)}")
            return None, None

    def log_wallet_balances(self):
        """Log all token balances in the wallet."""
        try:
            # Get ETH balance
            eth_balance = self.w3.eth.get_balance(self.account.address)
            eth_balance_in_ether = self.w3.from_wei(eth_balance, 'ether')
            logging.info(f"S Balance: {eth_balance_in_ether:.4f} S")

        except Exception as e:
            logging.error(f"Error logging wallet balances: {str(e)}")

    def ensure_pair_approvals(self) -> bool:
        """Check and approve both tokens if necessary."""

        if APPROVED_KEY in db:
            return True
        try:
            # Get token addresses from the pair contract
            logging.info("Fetching token addresses from pair contract")
            token_x_address = self.lbp_contract.functions.getTokenX().call()
            token_y_address = self.lbp_contract.functions.getTokenY().call()

            # Check current approvals
            logging.info("Checking and updating token approvals")

            # Check Token X
            symbol_x, allowance_x, is_approved_x = self.check_token_approval(
                token_x_address, LBROUTER_CONTRACT_ADDRESS)

            # Check Token Y
            symbol_y, allowance_y, is_approved_y = self.check_token_approval(
                token_y_address, LBROUTER_CONTRACT_ADDRESS)

            # Approve Token X if needed
            if not is_approved_x:
                logging.info(f"Approving {symbol_x}...")
                if not self.approve_token(token_x_address,
                                          LBROUTER_CONTRACT_ADDRESS):
                    return False

            # Approve Token Y if needed
            if not is_approved_y:
                logging.info(f"Approving {symbol_y}...")
                if not self.approve_token(token_y_address,
                                          LBROUTER_CONTRACT_ADDRESS):
                    return False

            # Update approval status in the database
            db[APPROVED_KEY] = True

            return True

        except Exception as e:
            logging.error(f"Error ensuring pair approvals: {str(e)}")
            return False

    def approve_token(self, token_address: str, spender_address: str) -> bool:
        """
        Approve maximum amount for a token
        Returns True if successful, False otherwise
        """
        try:
            token_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_address),
                abi=self.erc20_abi)

            # Max uint256 value (2^256 - 1)
            max_amount = (2**256) - 1

            logging.info("Building approval transaction")
            # Build approval transaction
            approve_tx = token_contract.functions.approve(
                spender_address, max_amount).build_transaction({
                    'from':
                    self.account.address,
                    'gas':
                    100000,  # Standard gas limit for approvals
                    'gasPrice':
                    self.w3.eth.gas_price,
                    'nonce':
                    self.w3.eth.get_transaction_count(self.account.address),
                })
            
            logging.info("Signing and sending approval transaction")

            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                approve_tx, self.account._private_key)
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_tx.raw_transaction)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            symbol = token_contract.functions.symbol().call()
            if receipt.status == 1:
                logging.info(f"Successfully approved {symbol} for router")
                return True
            else:
                logging.error(f"Failed to approve {symbol}")
                return False

        except Exception as e:
            logging.error(f"Error approving token {token_address}: {str(e)}")
            return False

    def calculate_bin_range(self, active_id: int) -> tuple:
        """Calculate the bin range based on the active bin."""
        range_start = active_id - BIN_RANGE
        range_end = active_id + BIN_RANGE
        return range_start, range_end

    def generate_bin_distribution(self, amount_x: int, amount_y: int) -> tuple:
        """
        Generate distribution arrays for multiple bins following DLMM mechanics:
        - Token X goes to bins right of active bin (higher price)
        - Token Y goes to bins left of active bin (lower price)
        - Active bin gets 50% of both tokens if both tokens are provided
        - All bins get equal distribution of their respective token
        """
        total_bins = 2 * BIN_RANGE + 1  # bins to the left + active bin + bins to the right
        middle_index = BIN_RANGE  # Index of the active bin in our arrays
        
        # Calculate total distribution (must sum to 10^18)
        total_distribution = 1000000000000000000  # 1e18 in decimal
        
        # Initialize distribution arrays
        distribution_x = [0] * total_bins
        distribution_y = [0] * total_bins
        
        # Calculate how much to allocate to active bin vs other bins
        active_bin_percent = 0.5  # 50% to active bin
        other_bins_percent = 1 - active_bin_percent  # 50% to other bins
        
        if amount_x > 0 and amount_y > 0:
            # Both tokens provided - distribute both across their respective sides
            # Active bin gets 50% of both tokens
            distribution_x[middle_index] = int(total_distribution * active_bin_percent)
            distribution_y[middle_index] = int(total_distribution * active_bin_percent)
            
            # Calculate per-bin distribution for remaining bins
            bins_right = BIN_RANGE
            bins_left = BIN_RANGE
            
            # Token X goes to bins to the right (higher price)
            if bins_right > 0:
                x_per_bin = int((total_distribution * other_bins_percent) / bins_right)
                for i in range(middle_index + 1, total_bins):
                    distribution_x[i] = x_per_bin
                # Adjust for rounding
                distribution_x[-1] += (total_distribution - distribution_x[middle_index] - 
                                    sum(distribution_x[middle_index+1:]))
            
            # Token Y goes to bins to the left (lower price)
            if bins_left > 0:
                y_per_bin = int((total_distribution * other_bins_percent) / bins_left)
                for i in range(0, middle_index):
                    distribution_y[i] = y_per_bin
                # Adjust for rounding
                distribution_y[0] += (total_distribution - distribution_y[middle_index] - 
                                    sum(distribution_y[:middle_index]))
                
        elif amount_x > 0:
            # Only X tokens provided
            # 50% to active bin, 50% to right bins
            distribution_x[middle_index] = int(total_distribution * active_bin_percent)
            
            bins_right = BIN_RANGE
            if bins_right > 0:
                x_per_bin = int((total_distribution * other_bins_percent) / bins_right)
                for i in range(middle_index + 1, total_bins):
                    distribution_x[i] = x_per_bin
                # Adjust for rounding
                distribution_x[-1] += (total_distribution - distribution_x[middle_index] - 
                                    sum(distribution_x[middle_index+1:]))
            else:
                # If no right bins, all goes to active bin
                distribution_x[middle_index] = total_distribution
                
        elif amount_y > 0:
            # Only Y tokens provided
            # 50% to active bin, 50% to left bins
            distribution_y[middle_index] = int(total_distribution * active_bin_percent)
            
            bins_left = BIN_RANGE
            if bins_left > 0:
                y_per_bin = int((total_distribution * other_bins_percent) / bins_left)
                for i in range(0, middle_index):
                    distribution_y[i] = y_per_bin
                # Adjust for rounding
                distribution_y[0] += (total_distribution - distribution_y[middle_index] - 
                                sum(distribution_y[:middle_index]))
            else:
                # If no left bins, all goes to active bin
                distribution_y[middle_index] = total_distribution
        
        # Verify distributions sum to exactly 10^18
        assert sum(distribution_x) == total_distribution, f"X distribution sums to {sum(distribution_x)} not {total_distribution}"
        assert sum(distribution_y) == total_distribution, f"Y distribution sums to {sum(distribution_y)} not {total_distribution}"
        
        return distribution_x, distribution_y

    def generate_delta_ids(self) -> list:
        """Generate delta IDs for the bin range."""
        # Create an array of deltaIds covering the bin range
        delta_ids = list(range(-BIN_RANGE, BIN_RANGE + 1))
        return delta_ids

    def swap_tokens(self, token_from: str, token_to: str, amount: int) -> int:
        """ Swap a portion of one token for another using the LBRouter Returns the amount of tokens received"""
        try:
            logging.info(f"Swapping {amount} of {token_from} for {token_to}")
            
            # Prepare path for swap
            pairBinSteps = [BIN_STEPS]
            versions = [2]  # Use version 1 for V2 pairs
            tokenPath = [token_from, token_to]
            
            # Build swap transaction
            deadline = int(datetime.now().timestamp()) + 60
            
            swap_tx = self.lbrouter_contract.functions.swapExactTokensForTokens(
                amount,
                0,
                [pairBinSteps, versions, tokenPath],
                self.account.address,
                deadline
            ).build_transaction({
                'from': self.account.address,
                'gas': 500000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(swap_tx, self.account._private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                # Get updated balance of token_to after swap
                _, new_balance = self.get_token_balance(token_to)
                logging.info(f"Successfully swapped to {new_balance} of {token_to}")
                return new_balance
            else:
                logging.error("Swap failed")
                return 0
                
        except Exception as e:
            logging.error(f"Error swapping tokens: {str(e)}")
            return 0
    
    def add_liquidity(self, active_id: int):
        """Add liquidity across a range of bins centered on the active bin."""
        try:
            # Check if there are any active positions
            if self.get_active_position():
                return True

            # Get token addresses
            token_x_address = self.lbp_contract.functions.getTokenX().call()
            token_y_address = self.lbp_contract.functions.getTokenY().call()

            # Get token balances and info
            symbol_x, balance_x_original = self.get_token_balance(token_x_address)
            symbol_y, balance_y_original = self.get_token_balance(token_y_address)

            # Get decimals
            tokenx_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_x_address), 
                abi=self.erc20_abi)
            decimals_x = tokenx_contract.functions.decimals().call()

            tokeny_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(token_y_address),
                abi=self.erc20_abi)
            decimals_y = tokeny_contract.functions.decimals().call()

            # Apply max funds constraints if set
            balance_x = balance_x_original
            balance_y = balance_y_original
            
            if USE_MAX_FUNDS_TOKENX > 0:
                if USE_MAX_FUNDS_TOKENX * (10**decimals_x) < balance_x:
                    balance_x = USE_MAX_FUNDS_TOKENX * (10**decimals_x)
                logging.info(f"Use max {symbol_x}: {balance_x / (10**decimals_x)}")

            if USE_MAX_FUNDS_TOKENY > 0:
                if USE_MAX_FUNDS_TOKENY * (10**decimals_y) < balance_y:
                    balance_y = USE_MAX_FUNDS_TOKENY * (10**decimals_y)
                logging.info(f"Use max {symbol_y}: {balance_y / (10**decimals_y)}")

            logging.info("Checking balances for liquidity:")
            logging.info(f"{symbol_x}: {balance_x / (10**decimals_x)}")
            logging.info(f"{symbol_y}: {balance_y / (10**decimals_y)}")

            # Check if we need to rebalance tokens
            has_x = balance_x > 0
            has_y = balance_y > 0
            
            # If we have only one token, swap half for the other
            if balance_x > 0 and balance_x > balance_y:
                # Split token X balance - keep half, swap half for Y
                amount_to_swap = balance_x // 2
                logging.info(f"Only {symbol_x} available. Swapping half ({amount_to_swap / (10**decimals_x)}) for {symbol_y}")
                
                # Perform the swap
                self.swap_tokens(token_x_address, token_y_address, amount_to_swap)
                
                # Update balances after swap
                _, balance_x = self.get_token_balance(token_x_address)
                _, balance_y = self.get_token_balance(token_y_address)
                
                logging.info(f"After swap: {symbol_x}: {balance_x / (10**decimals_x)}, {symbol_y}: {balance_y / (10**decimals_y)}")
                
            elif balance_y > 0 and balance_y > balance_x:
                # Split token Y balance - keep half, swap half for X
                amount_to_swap = balance_y // 2
                logging.info(f"Only {symbol_y} available. Swapping half ({amount_to_swap / (10**decimals_y)}) for {symbol_x}")
                
                # Perform the swap
                self.swap_tokens(token_y_address, token_x_address, amount_to_swap)
                
                # Update balances after swap
                _, balance_x = self.get_token_balance(token_x_address)
                _, balance_y = self.get_token_balance(token_y_address)
                
                logging.info(f"After swap: {symbol_x}: {balance_x / (10**decimals_x)}, {symbol_y}: {balance_y / (10**decimals_y)}")
            
            # Now both tokens should be available
            has_x = balance_x > 0
            has_y = balance_y > 0
            
            if balance_x < 1 and balance_y < 1:
                logging.error("No token balance available for liquidity even after swap attempt")
                return False
                
            # Calculate bin range
            range_start, range_end = self.calculate_bin_range(active_id)
            
            # Generate distribution arrays
            distribution_x, distribution_y = self.generate_bin_distribution(balance_x if has_x else 0, 
                                                                        balance_y if has_y else 0)
            
            # Generate delta IDs
            delta_ids = self.generate_delta_ids()
            
            # Log the distribution plan
            logging.info(f"Adding liquidity across bins {range_start} to {range_end}")
            logging.info(f"Active bin: {active_id}")
            logging.info(f"Total bins: {len(delta_ids)}")
            
            # The rest of the function remains the same
            # Prepare liquidity parameters
            add_params = {
                'tokenX': token_x_address,
                'tokenY': token_y_address,
                'binStep': BIN_STEPS,
                'amountX': balance_x if has_x else 0,
                'amountY': balance_y if has_y else 0,
                'amountXMin': 0,
                'amountYMin': 0,
                'activeIdDesired': active_id,
                'idSlippage': BIN_RANGE,  # Increased slippage to accommodate range
                'deltaIds': delta_ids,
                'distributionX': distribution_x,
                'distributionY': distribution_y,
                'refundTo': self.account.address,
                'to': self.account.address,  # owner's wallet
                'deadline': int(datetime.now().timestamp()) + 60  # Increased deadline
            }

            # Build transaction
            logging.info("Building transaction to add liquidity...")
            add_tx = self.lbrouter_contract.functions.addLiquidity(
                add_params).build_transaction({
                    'from': self.account.address,
                    'gas': 2000000,  # Increased gas limit for multiple bins
                    'gasPrice': self.w3.eth.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                })

            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(
                add_tx, self.account._private_key)
            tx_hash = self.w3.eth.send_raw_transaction(
                signed_tx.raw_transaction)

            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status == 1:
                # Generate bin IDs list for the position
                bin_ids = list(range(range_start, range_end + 1))
                
                # Add position to database
                new_position = LiquidityPosition(
                    bin_ids=bin_ids,
                    token_x=token_x_address,
                    token_y=token_y_address,
                    amount_x=balance_x if has_x else 0,
                    amount_y=balance_y if has_y else 0,
                    to_address=self.account.address,
                    range_start=range_start,
                    range_end=range_end
                )
                self.add_new_position(new_position)
                logging.info(f"Successfully added liquidity across {len(bin_ids)} bins")
                return True
            else:
                logging.error("Failed to add liquidity")
                return False

        except Exception as e:
            logging.error(f"Error adding liquidity: {str(e)}")
            return False
    
    def is_active_bin_in_range(self, active_id: int, position: LiquidityPosition) -> bool:
        """Check if the active bin is within our current position range."""
        return position.range_start <= active_id <= position.range_end

    def manage_liquidity(self):
        """Main function to manage liquidity positions across a range of bins."""
        # Ensure token approvals
        if not self.ensure_pair_approvals():
            logging.error(
                "Failed to ensure token approvals. Stopping liquidity management."
            )
            return
        try:
            # Get current active ID from contract
            logging.info("Getting current active bin from contract...")
            active_id = self.lbp_contract.functions.getActiveId().call()
            logging.info(f"Current active bin: {active_id}")

            # Try to add liquidity if no positions exist
            position = self.get_active_position()
            if not position:
                logging.info("No active position. Adding new liquidity...")
                self.add_liquidity(active_id)
                return

            # Check if active bin is within our current range
            if not self.is_active_bin_in_range(active_id, position):
                logging.info(
                    f"Active bin {active_id} is outside current range [{position.range_start}-{position.range_end}]. Rebalancing..."
                )
                
                # Withdraw existing liquidity
                if self.withdraw_liquidity(position):
                    del db[POSITION_KEY]
                    logging.info("Successfully withdrawn liquidity. Adding new liquidity...")
                    # Add new liquidity
                    self.add_liquidity(active_id)
                else:
                    logging.error("Failed to withdraw liquidity for rebalancing")
                    return
            else:
                logging.info(f"OK - Position is farming. Active bin {active_id} is within range [{position.range_start}-{position.range_end}]")

        except Exception as e:
            logging.error(f"Error in manage_liquidity: {str(e)}")


def main():
    logging.info("---------------------------------------")
    logging.info(f"Starting DLMM Liquidity Manager with bin range: {BIN_RANGE}")
    manager = LiquidityManager()
    manager.log_wallet_balances()
    while True:
        manager.manage_liquidity()
        time.sleep(30)
        logging.info("Rebalancing check...")


if __name__ == "__main__":
    main()