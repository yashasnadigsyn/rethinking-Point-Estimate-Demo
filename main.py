import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- New Distribution Generation Logic ---

def generate_distribution_params():
    """Chooses parameters for a Beta distribution to create varied shapes."""
    dist_category = np.random.choice(
        ['nearly_symmetric', 'moderately_skewed', 'strongly_skewed'],
        p=[0.2, 0.5, 0.3] # Adjust probabilities as desired (e.g., less symmetric)
    )

    if dist_category == 'nearly_symmetric':
        # Use large-ish, slightly different a, b to avoid exact 0.5 mean/median often
        base = np.random.randint(15, 30)
        diff = np.random.randint(1, 5)
        a = base + diff
        b = base
        if np.random.rand() > 0.5: # Randomly swap a and b
             a, b = b, a
        params = {'a': a, 'b': b, 'type': 'Nearly Symmetric'} # Internal type tracking
    elif dist_category == 'moderately_skewed':
        # One parameter moderately larger than the other
        low = np.random.randint(5, 10)
        high = np.random.randint(12, 25)
        if np.random.rand() > 0.5:
            a, b = low, high # Right skew
        else:
            a, b = high, low # Left skew
        params = {'a': a, 'b': b, 'type': 'Moderately Skewed'}
    else: # 'strongly_skewed'
        # One parameter small, the other large
        low = np.random.randint(2, 5)
        high = np.random.randint(10, 30)
        if np.random.rand() > 0.5:
            a, b = low, high # Strong right skew
        else:
            a, b = high, low # Strong left skew
        params = {'a': a, 'b': b, 'type': 'Strongly Skewed'}

    # Ensure a and b are at least 1 (required by np.random.beta)
    params['a'] = max(1, params['a'])
    params['b'] = max(1, params['b'])
    return params


def generate_samples(a, b, size=2000):
    """Generates samples from a Beta distribution."""
    return np.random.beta(a, b, size=size)

# --- Stat Calculation Functions ---
def calculate_mean(samples):
    return np.mean(samples)

def calculate_median(samples):
    return np.median(samples)

def calculate_mode(samples, bins=30):
    counts, bin_edges = np.histogram(samples, bins=bins, range=(0,1))
    if counts.size == 0: return 0.5
    max_bin = np.argmax(counts)
     # Check edge case where max_bin is the last bin index
    if max_bin == len(counts) -1:
         # For the last bin, maybe just take its left edge or center if possible
         # This simple avg might be outside [0,1] if bin_edges[max_bin+1] doesn't exist safely
         # Let's refine: take the middle of the max bin
         mode_estimate = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    else:
        mode_estimate = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    # Clamp mode estimate to [0, 1] just in case
    return np.clip(mode_estimate, 0.0, 1.0)


# --- Session State Initialization ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.round_number = 0
    st.session_state.remaining_money = 100.0

    # Generate distribution parameters and samples
    dist_params = generate_distribution_params()
    st.session_state.dist_params = dist_params # Store for potential later analysis/debug
    st.session_state.samples = generate_samples(dist_params['a'], dist_params['b'], size=2000)

    st.session_state.loss_function = np.random.choice(["absolute", "squared"])

    # The "Middle Point" IS the optimal point estimate for the chosen loss function
    if st.session_state.loss_function == "absolute":
        st.session_state.middle_point = calculate_median(st.session_state.samples)
        st.session_state.middle_point_type = "Median" # Keep for explanation logic
    else:  # squared loss
        st.session_state.middle_point = calculate_mean(st.session_state.samples)
        st.session_state.middle_point_type = "Mean" # Keep for explanation logic

    st.session_state.history = []
    st.session_state.game_over = False

# --- Main App ---
st.set_page_config(page_title="Find the Point Estimate!")
st.title("Find the Point Estimate!")
st.markdown("This game is inspired from the Chapter 3 *Sampling the Imaginary* of the book *Statistical Rethinking*. Given the entire posterior distribution, what value should you report? If I gave you a distribution and asked you to give me a single number that best represents the distribuion. Which value would you pick? Play the game below to understand it.")
st.markdown("---")

# --- Rules Displayed First ---
st.subheader("📜 Rules of the Game")
loss_formula_display = r"`Cost \propto |guess - target|`" if st.session_state.loss_function == "absolute" else r"`Cost \propto (guess - target)^2`"
st.markdown(f"""
- You start with **$100**.
- You will have 5 rounds. For each round, you will have to guess the point estimate of the shown distribution below.
- If your guess is **"exactly"** right, I won't deduct anything.
- But, for every wrong guess, I will deduct money from your savings based on the loss function.
- Example: If the loss is absolute, I will deduct |guess - true_value| and if it is squared, I will deduction (guess - true_value)^2.
- To maximize the winning, use the tools in the sidebar!
- All The Best!
""")
st.markdown("---") # Separator


# --- Sidebar ---
with st.sidebar:
    st.header("🛠️ Tools")
    if st.button("Calculate Mean"):
        mean_val = calculate_mean(st.session_state.samples)
        st.metric("Mean", f"{mean_val:.4f}")

    if st.button("Calculate Median"):
        median_val = calculate_median(st.session_state.samples)
        st.metric("Median", f"{median_val:.4f}")

    if st.button("Calculate Mode"):
        mode_val = calculate_mode(st.session_state.samples)
        st.metric("Mode", f"{mode_val:.4f}")

    st.markdown("---")
    st.header("📊 Game Status")
    st.metric("Current Round", f"{st.session_state.round_number + 1 if not st.session_state.game_over else 5}/5")
    st.metric("Your Savings", f"${st.session_state.remaining_money:.2f}")


# --- Main Area Continues ---

# Display Plot (without revealing type or target)
st.subheader("Distribution Shape")
fig_game, ax_game = plt.subplots()
sns.histplot(st.session_state.samples, kde=True, ax=ax_game, bins=40, stat="density") # Increased bins slightly
ax_game.set_title("Distribution of Values (0 to 1)") # Generic Title
ax_game.set_xlabel("Value")
ax_game.set_ylabel("Density")
ax_game.set_xlim(0, 1)
st.pyplot(fig_game)

# Display Loss Function Type (but not formula with target)
st.subheader("Loss Function Rule")
st.info(f"Your score penalty is calculated using a **{st.session_state.loss_function.capitalize()}** loss rule.")


# --- Game Logic ---
if not st.session_state.game_over:
    st.subheader(f"Round {st.session_state.round_number + 1} - Make Your Guess")

    guess = st.number_input(
        f"Enter your guess (0.000 to 1.000):",
        min_value=0.0, max_value=1.0, value=st.session_state.history[-1]['guess'] if st.session_state.history else 0.5, # Default to last guess or 0.5
        step=0.001, format="%.3f",
        key=f"guess_{st.session_state.round_number}",
        help="Use the tools in the sidebar to for help!"
    )

    submit_button = st.button("Submit Guess", key=f"submit_{st.session_state.round_number}")

    if submit_button:
        true_value = st.session_state.middle_point # This remains the hidden target
        loss = 0.0
        deduction = 0.0

        # Calculate loss based on the function
        if st.session_state.loss_function == "absolute":
            loss = np.abs(guess - true_value)
            deduction = 50 * loss
        else: # squared loss
            loss = (guess - true_value) ** 2
            deduction = 100 * loss

        # Ensure money doesn't go below zero
        new_remaining_money = max(0.0, st.session_state.remaining_money - deduction)
        actual_deduction = st.session_state.remaining_money - new_remaining_money

        # Store history (including internal type for explanation)
        st.session_state.history.append({
            'round': st.session_state.round_number + 1,
            'guess': guess,
            'target_point': true_value, # Store the target
            'target_type': st.session_state.middle_point_type, # Store Mean/Median
            'deduction': actual_deduction,
            'remaining_money_after': new_remaining_money
        })

        # Update game state
        st.session_state.remaining_money = new_remaining_money
        st.session_state.round_number += 1

        # Provide limited feedback for the round
        st.success(f"Round {st.session_state.round_number} Submitted!") # Use success/info
        st.metric(label="Money Deducted This Round", value=f"${actual_deduction:.2f}", delta=f"-${actual_deduction:.2f}", delta_color="inverse")
        # st.write(f"Remaining Money: ${st.session_state.remaining_money:.2f}") # This is shown in sidebar

        # Check game end conditions
        if st.session_state.round_number >= 5 or st.session_state.remaining_money <= 0:
            st.session_state.game_over = True
            if st.session_state.remaining_money <= 0 and st.session_state.round_number < 5:
                 st.warning("Oh no! You've run out of money. Game Over.")
            st.rerun() # Rerun to display Game Over section

        # Give a brief pause/confirmation before the next round input appears naturally on rerun
        import time
        time.sleep(0.1) # Short delay
        st.rerun() # Rerun to update round number display and input key


# --- Game Over Section ---
if st.session_state.game_over:
    st.subheader("🏁 Game Over! 🏁")
    st.success(f"You Saved: ${st.session_state.remaining_money:.2f}")

    # --- Final Plot with All Details ---
    st.subheader("Distribution Statistics")
    mean_val = calculate_mean(st.session_state.samples)
    median_val = calculate_median(st.session_state.samples)
    mode_val = calculate_mode(st.session_state.samples)

    fig_end, ax_end = plt.subplots()
    sns.histplot(st.session_state.samples, kde=True, ax=ax_end, bins=40, stat="density", label='Distribution Shape')
    ax_end.axvline(mean_val, color='orange', linestyle='--', label=f'Mean ({mean_val:.3f})', lw=2)
    ax_end.axvline(median_val, color='green', linestyle=':', label=f'Median ({median_val:.3f})', lw=2)
    ax_end.axvline(mode_val, color='purple', linestyle='-.', label=f'Mode ({mode_val:.3f})', lw=2)

    # Highlight the actual target point that was used
    target_val = st.session_state.middle_point
    target_type = st.session_state.middle_point_type
    target_color = 'green' if target_type == 'Median' else 'orange'
    ax_end.axvline(target_val, color='red', linestyle='-', label=f'ACTUAL TARGET ({target_type}): {target_val:.3f}', lw=3, alpha=0.8)

    ax_end.set_title("Distribution Statistics")
    ax_end.set_xlabel("Value")
    ax_end.set_ylabel("Density")
    ax_end.set_xlim(0, 1)
    ax_end.legend(fontsize='small')
    st.pyplot(fig_end)

    st.divider()

    # --- Explanation Based on Statistical Rethinking ---
    st.subheader("WTF is this?")
    # The explanation remains largely the same, as it reveals the concept after the game
    st.markdown(f"""
    You just played a game where the goal was to guess a point estimate of the distribution shown. 
    The key challenge was that the *best* guess depended on the **loss function** used to penalize errors.

    Choosing a single number (**point estimate**) to represent a distribution depends on how we define 'cost' or 'loss' function.
    
    To maximize the winnings (or minimize loss), we have to look at the loss function we are using.
                
    - Squared Loss(`(guess-target)^2`): Use mean.
    - Absolute Loss(`|guess-target|`): Use median.

    In this game, the loss function was **{st.session_state.loss_function.capitalize()}**. Therefore, the point estimate you were aiming for was the **{st.session_state.middle_point_type}**, which had a value of **{st.session_state.middle_point:.3f}**.

    Why Does This Matter? 

    So, why play a game about guessing a point based on a loss function? While Mean (for squared loss) and Median (for absolute loss) are standard "optimal" choices in statistics, this game illustrates a crucial concept from decision theory: **the best summary of information often depends entirely on the real-world consequences of being wrong.**

    The simple absolute and squared loss functions used here are just two possibilities. These below texts are completely taken from the book **Statistical Rethinking**.

    Real-world problems often have much more complex **loss functions**, reflecting asymmetric costs.

    Consider predicting hurricane wind speeds:
    -   **Underestimating** the speed could lead to catastrophic loss of life and property if evacuations aren't ordered. The cost is extremely high.
    -   **Overestimating** the speed might lead to unnecessary evacuation costs, which are significant but much lower than the cost of underestimation.

    In this scenario, the "loss" isn't symmetrical. The penalty for guessing too low is far greater than the penalty for guessing too high. For such an *asymmetric loss function*, the best single point estimate to guide decisions (like ordering an evacuation) wouldn't necessarily be the Mean or the Median. It would likely be a value *higher* than both, reflecting a bias towards caution to avoid the worst outcome.

    Furthermore, sometimes the goal isn't even to find the "best" point estimate of wind speed, but rather to make the optimal *decision* (evacuate or not?) directly, considering the probabilities from the entire distribution and the costs associated with each action.

    Choosing a single point estimate (like the Mean, Median, or Mode) inevitably discards information contained in the full distribution. While useful as summaries, it's vital to remember that their relevance depends on the *context* and the *specific purpose* (the implied loss function). Often, communicating the entire distribution (the full picture of uncertainty) is more valuable, allowing different people to apply their own relevant loss functions or make decisions directly. This game highlights how changing the "rules of the penalty" (the loss function) changes which summary point becomes the most strategic target.
    """)

    if st.button("Play Again?"):
        # Clear relevant session state keys to restart
        keys_to_clear = ['initialized', 'round_number', 'remaining_money', 'dist_params',
                         'samples', 'loss_function', 'middle_point', 'middle_point_type',
                         'history', 'game_over']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()