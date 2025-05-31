import streamlit as st
import re
from typing import Dict, Optional, List, Union


class StreamlitMathRenderer:
    """
    Ultra-robust math renderer for Streamlit applications.
    Handles LaTeX expressions, code blocks, and mixed content with extensive error handling.
    """
    
    def __init__(self):
        self.inline_patterns = [
            (r'\$([^$\n]+)\$', r'$\1$'),        # $...$ stays the same
            (r'\\\(([^)]+)\\\)', r'$\1$')       # \(...\) becomes $...$
        ]
        self.display_patterns = [
            r'(\$\$.*?\$\$)',                   # $$...$$
            r'(\\\[.*?\\\])'                    # \[...\]
        ]
    
    def _safe_str(self, text: Union[str, None, any]) -> str:
        """Safely convert any input to string, handling None and other types."""
        if text is None:
            return ""
        if not isinstance(text, str):
            try:
                return str(text)
            except:
                return ""
        return text
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for duplicate comparison."""
        if not text:
            return ""
        
        # Remove unicode formatting characters
        text = re.sub(r'[\u200b-\u200f\u2060\ufeff]', '', text)
        # Normalize multiple spaces to single space
        text = re.sub(r'\s+', ' ', text)
        # Remove common math formatting variations (subscript numbers)
        text = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]', lambda m: str(ord(m.group()) - ord('₀')), text)
        
        return text.strip()
    
    def _remove_line_duplicates(self, line: str) -> str:
        """Remove duplicates within a single line."""
        if not line or len(line) < 10:  # Skip very short lines
            return line
        
        # Try to find repeated patterns
        # Look for patterns that are at least 10 characters and repeated
        for length in range(len(line) // 2, 9, -1):  # Start from half length, go down to 10
            for start in range(len(line) - length * 2 + 1):
                pattern = line[start:start + length]
                
                # Check if this pattern repeats immediately after
                next_pos = start + length
                if next_pos + length <= len(line):
                    next_pattern = line[next_pos:next_pos + length]
                    
                    # Normalize both patterns for comparison
                    pattern_norm = self._normalize_for_comparison(pattern)
                    next_pattern_norm = self._normalize_for_comparison(next_pattern)
                    
                    if pattern_norm == next_pattern_norm and len(pattern_norm) > 5:
                        # Found a duplicate, remove the second occurrence
                        return line[:next_pos] + line[next_pos + length:]
        
        return line
    
    def _remove_inline_duplicates(self, text: str) -> str:
        """Remove duplicated content within the same line."""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = self._remove_line_duplicates(line)
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning that handles inline duplicates and variations."""
        if not text:
            return ""
        
        # First, handle inline duplicates (repeated content on same line)
        text = self._remove_inline_duplicates(text)
        
        # Then handle line-by-line duplicates
        lines = text.split('\n')
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip()
            # Normalize the line for comparison (remove extra spaces, unicode variations)
            line_normalized = self._normalize_for_comparison(line_clean)
            
            if line_clean and line_normalized not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line_normalized)
            elif not line_clean:  # Keep empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def detect_math_expressions(self, text: str) -> Dict[str, any]:
        """
        Detect various math expression patterns in text.
        Avoids false positives from code blocks and programming syntax.
        """
        text = self._safe_str(text)
        if not text:
            return {'has_math': False, 'inline_count': 0, 'display_count': 0, 'bracket_count': 0}

        try:
            # Clean the text first
            text = self._clean_text(text)
            
            # Remove code blocks to avoid false math detection
            text_no_code = self._remove_code_blocks(text)

            # Count inline math expressions with single $
            inline_dollar = len(re.findall(r'\$(?!\$)([^$\n]*[a-zA-Z_{}\\^][^$\n]*)\$', text_no_code))

            # Count display math with double $$...$$
            display_dollar = len(re.findall(r'\$\$.*?[a-zA-Z_{}\\^]+.*?\$\$', text_no_code, re.DOTALL))

            # Count math in LaTeX-like \( ... \) syntax
            bracket_math = len(re.findall(r'\\\(([^)]*?[a-zA-Z_{}\\^]+[^)]*?)\\\)', text_no_code))

            # Count \[...\] display math
            display_bracket = len(re.findall(r'\\\[.*?[a-zA-Z_{}\\^]+.*?\\\]', text_no_code, re.DOTALL))

            has_math = inline_dollar > 0 or display_dollar > 0 or bracket_math > 0 or display_bracket > 0

            return {
                'has_math': has_math,
                'inline_count': inline_dollar,
                'display_count': display_dollar + display_bracket,
                'bracket_count': bracket_math
            }
        except Exception as e:
            # If detection fails, assume no math
            return {'has_math': False, 'inline_count': 0, 'display_count': 0, 'bracket_count': 0}
    
    def _remove_code_blocks(self, text: str) -> str:
        """Remove code blocks to avoid false math detection."""
        text = self._safe_str(text)
        if not text:
            return ""
        
        try:
            # Remove triple backtick code blocks
            text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            # Remove single backtick inline code
            text_no_code = re.sub(r'`[^`]*`', '', text_no_code)
            return text_no_code
        except Exception:
            return text
    
    def render_inline_math(self, text: str) -> None:
        """Render text with inline math expressions."""
        text = self._safe_str(text)
        if not text:
            return
        
        try:
            processed_text = text
            for pattern, replacement in self.inline_patterns:
                try:
                    processed_text = re.sub(pattern, replacement, processed_text)
                except Exception:
                    continue  # Skip problematic patterns
            
            # Render with markdown (Streamlit handles $...$ inline math)
            st.markdown(processed_text)
        except Exception as e:
            # Fallback to plain text
            st.text(text)
    
    def render_display_math(self, math_content: Union[str, None]) -> None:
        """Render display math content with robust error handling."""
        math_content = self._safe_str(math_content)
        
        if not math_content or not math_content.strip():
            return
        
        try:
            # Clean the math content
            cleaned_content = math_content.strip()
            
            # Try to render with st.latex
            st.latex(cleaned_content)
        except Exception as e:
            try:
                # Fallback: try with markdown math
                st.markdown(f"$${cleaned_content}$$")
            except Exception:
                # Final fallback: render as code
                st.code(f"Math expression: {cleaned_content}")
    
    def process_mixed_content(self, text: str) -> None:
        """Process text with mixed content including display and inline math."""
        text = self._safe_str(text)
        if not text:
            return
        
        try:
            # Clean text first
            text = self._clean_text(text)
            
            # Combine all display math patterns
            combined_pattern = '|'.join(self.display_patterns)
            parts = re.split(f'({combined_pattern})', text, flags=re.DOTALL)
            
            for part in parts:
                part = self._safe_str(part)
                if not part or not part.strip():
                    continue
                    
                # Check if this part is display math
                if self._is_display_math(part):
                    math_content = self._extract_math_content(part)
                    self.render_display_math(math_content)
                else:
                    # Handle inline math and regular text
                    self.render_inline_math(part)
        except Exception as e:
            # Fallback to simple markdown
            st.markdown(text)
    
    def _is_display_math(self, text: str) -> bool:
        """Check if text is display math."""
        text = self._safe_str(text)
        if not text:
            return False
        
        try:
            return ((text.startswith('$$') and text.endswith('$$')) or
                    (text.startswith('\\[') and text.endswith('\\]')))
        except Exception:
            return False
    
    def _extract_math_content(self, text: str) -> str:
        """Extract math content from display math delimiters."""
        text = self._safe_str(text)
        if not text:
            return ""
        
        try:
            if text.startswith('$$') and text.endswith('$$') and len(text) > 4:
                return text[2:-2]
            elif text.startswith('\\[') and text.endswith('\\]') and len(text) > 4:
                return text[2:-2]
            return text
        except Exception:
            return text
    
    def process_multiline_response(self, response_text: str) -> None:
        """
        Process multi-line LLM response with enhanced math handling.
        Handles bracket notation, multi-line expressions, and mixed content.
        """
        response_text = self._safe_str(response_text)
        if not response_text:
            return
        
        try:
            # Clean the response first
            response_text = self._clean_text(response_text)
            
            lines = response_text.split('\n')
            i = 0
            
            while i < len(lines):
                line = self._safe_str(lines[i]) if i < len(lines) else ""
                
                if not line.strip():
                    st.write("")  # Empty line
                    i += 1
                    continue
                
                # Check for display math in brackets [ formula ]
                if self._is_bracket_math(line):
                    math_content = self._extract_bracket_math(line)
                    self.render_display_math(math_content)
                    i += 1
                    continue
                
                # Handle multi-line display math
                try:
                    math_block, end_index = self._extract_multiline_math(lines, i)
                    if math_block is not None:
                        self.render_display_math(math_block)
                        i = end_index + 1
                        continue
                except Exception:
                    pass  # Continue with regular processing
                
                # Regular text with possible inline math
                self.process_mixed_content(line)
                i += 1
        except Exception as e:
            # Final fallback
            st.markdown(response_text)
    
    def _is_bracket_math(self, line: str) -> bool:
        """Check if line contains bracket notation math."""
        line = self._safe_str(line)
        if not line:
            return False
        
        try:
            bracket_math_pattern = r'^\s*\[\s*([^[\]]*[a-zA-Z_{}\\]+[^[\]]*)\s*\]\s*$'
            bracket_match = re.match(bracket_math_pattern, line)
            return bracket_match and re.search(r'[a-zA-Z_{}\\]', line)
        except Exception:
            return False
    
    def _extract_bracket_math(self, line: str) -> str:
        """Extract math content from bracket notation."""
        line = self._safe_str(line)
        if not line:
            return ""
        
        try:
            math_content = re.sub(r'^\s*\[\s*', '', line)
            math_content = re.sub(r'\s*\]\s*$', '', math_content)
            return math_content
        except Exception:
            return line
    
    def _extract_multiline_math(self, lines: List[str], start_index: int) -> tuple:
        """Extract multi-line math blocks."""
        if start_index >= len(lines):
            return None, start_index
        
        try:
            line = self._safe_str(lines[start_index])
            if not line:
                return None, start_index
            
            # Check for multi-line $$ blocks
            if line.strip().startswith('$$') and not line.strip().endswith('$$'):
                return self._extract_multiline_delimiter(lines, start_index, '$$', '$$')
            
            # Check for multi-line \[ blocks
            if line.strip().startswith('\\[') and not line.strip().endswith('\\]'):
                return self._extract_multiline_delimiter(lines, start_index, '\\[', '\\]')
            
            return None, start_index
        except Exception:
            return None, start_index
    
    def _extract_multiline_delimiter(self, lines: List[str], start_index: int, 
                                   start_delim: str, end_delim: str) -> tuple:
        """Extract content between multi-line delimiters."""
        if start_index >= len(lines):
            return None, start_index
        
        try:
            first_line = self._safe_str(lines[start_index]).strip()
            if len(first_line) <= len(start_delim):
                math_lines = [""]
            else:
                math_lines = [first_line[len(start_delim):]]
            
            i = start_index + 1
            
            while i < len(lines):
                current_line = self._safe_str(lines[i]) if i < len(lines) else ""
                if current_line.strip().endswith(end_delim):
                    if len(current_line.strip()) > len(end_delim):
                        math_lines.append(current_line.strip()[:-len(end_delim)])
                    else:
                        math_lines.append("")
                    break
                else:
                    math_lines.append(current_line)
                i += 1
            else:
                # No closing delimiter found
                return None, start_index
            
            math_content = '\n'.join(math_lines).strip()
            return math_content if math_content else None, i
        except Exception:
            return None, start_index
    
    def render_with_code_preservation(self, text: str) -> None:
        """Render text while preserving code blocks."""
        text = self._safe_str(text)
        if not text:
            return
        
        try:
            # Split by code blocks (both ``` and ` varieties)
            parts = re.split(r'(```[\s\S]*?```|`[^`]*`)', text)
            
            for part in parts:
                part = self._safe_str(part)
                if not part:
                    continue
                elif part.startswith('```') and part.endswith('```'):
                    self._render_code_block(part)
                elif part.startswith('`') and part.endswith('`'):
                    self._render_inline_code(part)
                else:
                    # Regular text - check for math
                    math_info = self.detect_math_expressions(part)
                    if math_info['has_math']:
                        self.process_multiline_response(part)
                    else:
                        st.markdown(part)
        except Exception as e:
            # Fallback
            st.markdown(text)
    
    def _render_code_block(self, code_block: str) -> None:
        """Render a code block with language detection."""
        code_block = self._safe_str(code_block)
        if not code_block or len(code_block) < 6:  # Minimum for ```x```
            return
        
        try:
            code_content = code_block[3:-3]
            lines = code_content.split('\n')
            
            if lines and lines[0] and lines[0].strip() and ' ' not in lines[0].strip():
                language = lines[0].strip()
                code_content = '\n'.join(lines[1:])
                st.code(code_content, language=language)
            else:
                st.code(code_content)
        except Exception:
            st.text(code_block)
    
    def _render_inline_code(self, code: str) -> None:
        """Render inline code."""
        code = self._safe_str(code)
        if not code or len(code) < 2:
            return
        
        try:
            st.code(code[1:-1], language=None)
        except Exception:
            st.text(code)
    
    def smart_render(self, text: str) -> None:
        """
        Main entry point for intelligent math rendering.
        Automatically detects content type and renders appropriately.
        """
        text = self._safe_str(text)
        if not text:
            return
        
        try:
            # Check if text contains code blocks
            has_code_blocks = bool(re.search(r'```|`[^`]+`', text))
            
            if has_code_blocks:
                self.render_with_code_preservation(text)
            else:
                # Detect what types of math we're dealing with
                math_info = self.detect_math_expressions(text)
                
                if not math_info['has_math']:
                    # No math detected, render as regular markdown
                    st.markdown(text)
                    return
                
                # Has math, use the comprehensive processor
                self.process_multiline_response(text)
        except Exception as e:
            # Ultimate fallback
            st.markdown(f"**Content:** {text}")


# Convenience functions for backward compatibility and ease of use
def render_math_content(text: str) -> None:
    """Render text containing LaTeX math expressions in Streamlit."""
    renderer = StreamlitMathRenderer()
    renderer.process_mixed_content(text)


def process_llm_response(response_text: str) -> None:
    """Process LLM response and render it with proper math formatting."""
    renderer = StreamlitMathRenderer()
    renderer.process_multiline_response(response_text)


def smart_math_renderer(text: str) -> None:
    """Intelligent math renderer with automatic content detection."""
    renderer = StreamlitMathRenderer()
    renderer.smart_render(text)


def detect_math_expressions(text: str) -> Dict[str, any]:
    """Utility function to detect math expressions in text."""
    renderer = StreamlitMathRenderer()
    return renderer.detect_math_expressions(text)
