import pygame

class ScrollableMarkdownArea:
    """Area with scrollable markdown content."""
    
    def __init__(self, x, y, width, height, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.screen = screen
        self.scroll_offset = 0
        self.content_height = 0
        self.max_scroll = 0
        self.scroll_speed = 20
        
    def handle_scroll(self, event):
        """Handle mouse wheel scroll events."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - self.scroll_speed)
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + self.scroll_speed)
    
    def render_markdown(self, markdown_text):
        """Render markdown text with scroll support."""
        # Cria uma surface temporária para desenhar todo o conteúdo
        temp_surface = pygame.Surface((self.width, max(self.height, 2000)))
        temp_surface.fill((245, 245, 245))
        
        text_y = 10
        lines = markdown_text.split('\n')
        
        for line in lines:
            if text_y > 2000 - 50:  # Limite de segurança
                break
            increment = render_markdown_line(line, text_y, temp_surface, base_x=10, max_width=self.width - 40)
            text_y += increment
        
        self.content_height = text_y + 10
        self.max_scroll = max(0, self.content_height - self.height)
        
        # Desenha a área visível com scroll
        # Fundo
        pygame.draw.rect(self.screen, (245, 245, 245), (self.x, self.y, self.width, self.height))
        pygame.draw.line(self.screen, (0, 0, 0), (self.x, self.y), (self.x + self.width, self.y), 3)
        
        # Blit apenas a parte visível
        visible_rect = pygame.Rect(0, self.scroll_offset, self.width, self.height)
        self.screen.blit(temp_surface, (self.x, self.y), visible_rect)
        
        # Desenha scrollbar se necessário
        if self.content_height > self.height:
            self._draw_scrollbar()
    
    def _draw_scrollbar(self):
        """Draw a scrollbar on the right side."""
        scrollbar_width = 10
        scrollbar_x = self.x + self.width - scrollbar_width - 5
        scrollbar_y = self.y + 5
        scrollbar_height = self.height - 10
        
        # Fundo da scrollbar
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (scrollbar_x, scrollbar_y, scrollbar_width, scrollbar_height))
        
        # Thumb da scrollbar
        thumb_height = max(20, int(scrollbar_height * (self.height / self.content_height)))
        thumb_y = scrollbar_y + int((self.scroll_offset / self.max_scroll) * (scrollbar_height - thumb_height)) if self.max_scroll > 0 else scrollbar_y
        
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (scrollbar_x, thumb_y, scrollbar_width, thumb_height))

    
def render_markdown_line(line, y_pos, screen, base_x=20, max_width=1160):
    """Render a single line of markdown-like text with word wrapping."""
    import re
    
    # Fontes
    regular_font = pygame.font.SysFont("Arial", 12)
    bold_font = pygame.font.SysFont("Arial", 12, bold=True)
    header_font = pygame.font.SysFont("Arial", 14, bold=True)
    
    line = line.strip()
    if not line:
        return 15
    
    x_pos = base_x
    line_height = 18
    current_y = y_pos
    total_height = 0
    
    # Headers (## ou ###)
    if line.startswith('###'):
        text = line[3:].strip()
        surface = pygame.font.SysFont("Arial", 14, bold=True).render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height
    elif line.startswith('##'):
        text = line[2:].strip()
        surface = header_font.render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height + 2
    elif line.startswith('#'):
        text = line[1:].strip()
        surface = pygame.font.SysFont("Arial", 18, bold=True).render(text, True, (0, 0, 0))
        screen.blit(surface, (x_pos, current_y))
        return line_height + 5
    
    # Lista com marcadores
    indent = 0
    if line.startswith('- ') or line.startswith('* '):
        bullet_surface = regular_font.render("•", True, (0, 0, 0))
        screen.blit(bullet_surface, (x_pos, current_y))
        indent = 15
        x_pos += indent
        line = line[2:].strip()
    
    # Processa negrito (**texto**) e quebra de linha
    parts = re.split(r'(\*\*.*?\*\*)', line)
    
    for part in parts:
        if not part:
            continue
            
        is_bold = part.startswith('**') and part.endswith('**')
        text = part[2:-2] if is_bold else part
        font = bold_font if is_bold else regular_font
        
        # Quebra o texto em palavras para word wrapping
        words = text.split(' ')
        for i, word in enumerate(words):
            # Adiciona espaço antes da palavra (exceto a primeira)
            if i > 0:
                word = ' ' + word
            
            word_surface = font.render(word, True, (0, 0, 0))
            word_width = word_surface.get_width()
            
            # Verifica se precisa quebrar linha
            if x_pos + word_width > base_x + max_width:
                current_y += line_height
                total_height += line_height
                x_pos = base_x + indent  # Mantém indentação em linhas subsequentes
            
            screen.blit(word_surface, (x_pos, current_y))
            x_pos += word_width
    
    return line_height + total_height