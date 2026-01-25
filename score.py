class Fighter:
    def __init__(self, name):
        self.name = name
        self.strike = 0
        self.takedown = 0
        self.knockdown = 0
        self.score = 0

    def update_score(self):
        """Calculate total score from actions"""
        self.score = (self.strike * 10 + 
                      self.takedown * 15 + 
                      self.knockdown * 20)

    def __repr__(self):
        return f"{self.name}: {self.score}"


def get_score(fighters):
    """Return formatted score display for all fighters"""
    score_text = " | ".join([f"{f.name}: {f.score}" for f in fighters])
    return score_text


def get_winner(fighters):
    """Determine and return winner(s)"""
    if len(fighters) == 2:
        if fighters[0].score > fighters[1].score:
            return f"{fighters[0].name} wins {fighters[0].score}-{fighters[1].score}"
        elif fighters[1].score > fighters[0].score:
            return f"{fighters[1].name} wins {fighters[1].score}-{fighters[0].score}"
        else:
            return "Draw"
    
    elif len(fighters) == 3:
        # sort by score descending
        sorted_fighters = sorted(fighters, key=lambda f: f.score, reverse=True)
        
        if sorted_fighters[0].score > sorted_fighters[1].score:
            return (f"Winner: {sorted_fighters[0].name} ({sorted_fighters[0].score}) | "
                   f"2nd: {sorted_fighters[1].name} ({sorted_fighters[1].score}) | "
                   f"3rd: {sorted_fighters[2].name} ({sorted_fighters[2].score})")
        else:
            return (f"Tie between {sorted_fighters[0].name} and {sorted_fighters[1].name} "
                   f"({sorted_fighters[0].score}) | 3rd: {sorted_fighters[2].name} ({sorted_fighters[2].score})")