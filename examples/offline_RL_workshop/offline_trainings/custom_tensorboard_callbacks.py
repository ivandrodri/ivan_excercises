from tensorboardX import SummaryWriter


class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, env):
        super().__init__(log_dir)
        self.env = env
        self.add_text("args", "ToDo: Add much moreeeeeeeee options here !!")

    def log_custom_info(self):
        # Access environment information, e.g., env.state
        #info = self.env.max_num_steps
        info = 3
        # Log custom information from the environment
        self.add_scalar("custom_info/max_num_steps", info)
        #info_2 = self.env.nrow
        info_2 = 12.3
        self.add_scalar("custom_info/nrow", info_2)


